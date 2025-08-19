import json
import re
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from transformers import pipeline
from models import Candidate
from config import HUGGINGFACE_API_TOKEN
from embeddings_manager import embeddings_manager
import logging

# Set cache directory to prevent re-downloads
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './hf_cache'

# Suppress verbose logging from transformers and sentence-transformers
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress batch progress bars from sentence-transformers
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

class AIResumeParser:
    """AI-powered resume parser using LangChain and HuggingFace"""
    
    def __init__(self):
        # Initialize code generation model for structured output
        try:
            logger.info("Loading CodeT5-small model...")
            # Use CodeT5 - specifically designed for code/JSON generation
            self.text_generator = pipeline(
                "text2text-generation",
                model="Salesforce/codet5-small",
                max_length=256,
                do_sample=False,  # Deterministic output
                device=-1  # Use CPU to avoid GPU issues
            )
            logger.info("✅ Code generation model (CodeT5-small) loaded successfully")
        except Exception as e:
            logger.error(f"❌ Could not load AI model: {e}")
            logger.info("📋 Using fallback extraction mode")
            self.text_generator = None
        
        # Initialize text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Extraction prompt template
        self.extraction_prompt = PromptTemplate(
            input_variables=["resume_text"],
            template="""
You are an expert HR assistant. Extract the following information from this resume text and return it as valid JSON:

Resume Text:
{resume_text}

Extract and return ONLY valid JSON in this exact format:
{{
    "name": "Full Name",
    "email": "email@example.com or null",
    "phone": "phone number or null", 
    "skills": ["skill1", "skill2", "skill3"],
    "experience_years": number,
    "current_role": "job title or null"
}}

Rules:
1. For skills, extract technical skills, programming languages, frameworks, tools
2. For experience_years, calculate total years of professional experience
3. Return only the JSON, no other text
4. Use null for missing information, not empty strings

JSON:"""
        )
    
    def load_pdf_with_langchain(self, pdf_path: str) -> str:
        """Load PDF using LangChain document loader"""
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Combine all pages
            full_text = ""
            for i, doc in enumerate(documents):
                page_content = doc.page_content
                full_text += page_content + "\n"
                logger.debug(f"Page {i+1} content length: {len(page_content)} chars")
            
            # Log extraction details
            logger.info(f"📄 PDF Extraction Summary:")
            logger.info(f"   • Pages: {len(documents)}")
            logger.info(f"   • Total text length: {len(full_text)} characters")
            logger.info(f"   • Contains @ (email): {'@' in full_text}")
            
            # Check for phone patterns (moved outside f-string)
            phone_pattern = r'\d{3}[-.]?\d{3}[-.]?\d{4}'
            has_phone = bool(re.search(phone_pattern, full_text))
            logger.info(f"   • Contains phone patterns: {has_phone}")
            
            # Show sample of extracted text
            sample_text = full_text[:500].replace('\n', ' ').strip()
            logger.info(f"📝 First 500 chars: '{sample_text}...'")
            
            if len(full_text.strip()) == 0:
                logger.warning("⚠️  WARNING: Extracted text is empty!")
                
            return full_text
            
        except Exception as e:
            logger.error(f"❌ Error loading PDF with LangChain: {e}")
            raise
    
    def chunk_text_with_langchain(self, text: str) -> List[str]:
        """Split text into chunks using LangChain text splitter"""
        docs = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]
    
    def extract_with_ai(self, text: str) -> Dict[str, Any]:
        """Extract information using AI model (CodeT5)"""
        if not self.text_generator:
            self.extraction_method = "Fallback (No AI Model)"
            return self.fallback_extraction(text)
        
        try:
            # Limit text size for model processing
            text_sample = text[:800]  # Reduce for better processing
            
            # Log what the AI will actually process
            logger.info(f"🤖 AI Input Summary:")
            logger.info(f"   • Text sample length: {len(text_sample)} chars")
            # Check for name patterns (moved outside f-string)
            name_pattern = r'[A-Z][a-z]+ [A-Z][a-z]+'
            has_name_pattern = bool(re.search(name_pattern, text_sample))
            logger.info(f"   • Sample contains name-like patterns: {has_name_pattern}")
            logger.info(f"   • Sample text: '{text_sample[:200].replace(chr(10), ' ').strip()}...'")
            
            # Create instruction prompt without template
            prompt = f"""Extract the following from this resume and output as JSON:
- name (the person's full name)
- email (email address if present)
- phone (phone number if present)
- skills (array of technical skills mentioned)
- experience_years (total years of work experience as a number)
- current_role (current job title if mentioned)

Resume:
{text_sample[:500]}

Output JSON:"""
            
            # Generate response with CodeT5
            logger.info("Generating AI response with CodeT5...")
            response = self.text_generator(
                prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.1,  # More deterministic
                pad_token_id=self.text_generator.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            logger.info(f"AI generated: {generated_text[:200]}...")
            
            # Log the raw response for debugging
            logger.debug(f"Full AI response: {generated_text}")
            
            # Extract only the generated content (after the prompt)
            # CodeT5 may include the prompt in output, so remove it
            if prompt in generated_text:
                generated_content = generated_text[len(prompt):]
            else:
                generated_content = generated_text
            
            # Clean up the response
            full_json = generated_content.strip()
            
            # Try to find and parse JSON
            json_start = full_json.find('{')
            json_end = full_json.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = full_json[json_start:json_end]
                
                # Clean up common CodeT5 artifacts
                json_str = json_str.replace(':::', '').replace('...', '').strip()
                
                try:
                    extracted_data = json.loads(json_str)
                    logger.info("AI extraction successful")
                    
                    # Validate extracted data structure
                    if not isinstance(extracted_data, dict):
                        raise ValueError("Generated data is not a dictionary")
                    
                    # Ensure required fields with defaults
                    extracted_data = {
                        "name": extracted_data.get("name", "Unknown"),
                        "email": extracted_data.get("email"),
                        "phone": extracted_data.get("phone"),
                        "skills": extracted_data.get("skills", []),
                        "experience_years": extracted_data.get("experience_years", 0),
                        "current_role": extracted_data.get("current_role")
                    }
                    
                    # Enhance with semantic skill extraction
                    semantic_skills = embeddings_manager.extract_skills_semantic(text, threshold=0.7)
                    
                    if extracted_data.get('skills') and isinstance(extracted_data['skills'], list):
                        # Combine AI skills with semantic skills
                        ai_skills = extracted_data['skills']
                        all_skills = list(set(ai_skills + semantic_skills))
                    else:
                        # Use only semantic skills if AI didn't extract skills properly
                        all_skills = semantic_skills
                    
                    extracted_data['skills'] = embeddings_manager.normalize_skills(all_skills)
                    logger.info(f"Enhanced skills with embeddings: {len(extracted_data['skills'])} total")
                    
                    self.extraction_method = "AI + Embeddings"
                    return extracted_data
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Invalid JSON from AI ({e}), using fallback")
                    self.extraction_method = "Fallback (AI JSON Error)"
                    return self.fallback_extraction(text)
            else:
                logger.warning("No JSON found in AI response, using fallback")
                self.extraction_method = "Fallback (No AI JSON)"
                return self.fallback_extraction(text)
                
        except Exception as e:
            logger.error(f"AI extraction failed: {e}, using fallback")
            self.extraction_method = "Fallback (AI Error)"
            return self.fallback_extraction(text)
    
    def fallback_extraction(self, text: str) -> Dict[str, Any]:
        """Fallback regex extraction if AI fails"""
        logger.info("Using fallback regex extraction")
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        email = email_match.group(0) if email_match else None
        
        # Phone extraction
        phone_patterns = [
            r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',
            r'\+[0-9]{1,3}[\s.-]?[0-9]{4,14}',
        ]
        phone = None
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(0).strip()
                break
        
        # Name extraction (first clean line)
        lines = text.split('\n')
        name = "Unknown"
        for line in lines[:10]:
            line = line.strip()
            if (line and len(line) < 50 and 
                not '@' in line and not '|' in line and 
                re.match(r'^[A-Za-z\s\-\.]+$', line)):
                words = line.split()
                if 2 <= len(words) <= 4:
                    name = line
                    break
        
        # Skills extraction using semantic embeddings
        # First try semantic extraction
        semantic_skills = embeddings_manager.extract_skills_semantic(text, threshold=0.7)
        
        # Fallback: Basic keyword extraction
        if not semantic_skills:
            common_skills = [
                "Python", "Java", "JavaScript", "TypeScript", "React", "Angular", 
                "Vue", "Node.js", "Django", "FastAPI", "Flask", "SQL", "PostgreSQL", 
                "MongoDB", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git",
                "HTML", "CSS", "C++", "C#", ".NET", "Spring", "Linux"
            ]
            
            found_skills = []
            text_lower = text.lower()
            for skill in common_skills:
                if skill.lower() in text_lower:
                    found_skills.append(skill)
        else:
            # Normalize skills to canonical forms
            found_skills = embeddings_manager.normalize_skills(semantic_skills)
            logger.info(f"Extracted {len(found_skills)} skills using semantic embeddings")
        
        # Experience extraction - improved patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|with|of|working)',
            r'(\d+)\s*years?\s*experience',  # Simple "2 years experience"
            r'(\d+)\s*year',  # Even simpler "2 year"
            r'over\s*(\d+)\s*years?',
            r'(\d+)\s*years?\s*(?:in\s*)?(?:software|development|programming|tech)',
            r'total\s*(?:of\s*)?(\d+)\s*years?',
        ]
        experience_years = 0
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                experience_years = int(match.group(1))
                logger.info(f"Found experience: {experience_years} years using pattern: {pattern}")
                break
        
        # Also check for date ranges like 2022-2024
        if experience_years == 0:
            date_pattern = r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)'
            date_matches = re.findall(date_pattern, text, re.IGNORECASE)
            if date_matches:
                for start_year, end_year in date_matches:
                    end = 2024 if end_year.lower() in ['present', 'current'] else int(end_year)
                    years_diff = end - int(start_year)
                    if years_diff > experience_years:
                        experience_years = years_diff
                        logger.info(f"Calculated experience from dates: {experience_years} years")
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "skills": found_skills,
            "experience_years": experience_years,
            "current_role": None
        }
    
    def validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted data"""
        # Ensure required fields exist
        if not data.get("name") or data["name"] == "Unknown":
            # Try to extract from email if available
            if data.get("email"):
                email_name = data["email"].split("@")[0]
                data["name"] = email_name.replace(".", " ").title()
            else:
                data["name"] = "Unknown Candidate"
        
        # Validate email format
        if data.get("email"):
            email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
            if not re.match(email_pattern, data["email"]):
                data["email"] = None
        
        # Ensure skills is a list
        if not isinstance(data.get("skills"), list):
            data["skills"] = []
        
        # Validate experience years
        if not isinstance(data.get("experience_years"), int):
            data["experience_years"] = 0
        if data["experience_years"] < 0 or data["experience_years"] > 50:
            data["experience_years"] = 0
        
        return data
    
    def parse_resume(self, pdf_path: str) -> tuple[Candidate, str]:
        """Main method to parse resume using AI and return Candidate object"""
        try:
            logger.info(f"Starting AI-powered resume parsing for: {pdf_path}")
            
            # Step 1: Load PDF using LangChain
            logger.info("Loading PDF...")
            text = self.load_pdf_with_langchain(pdf_path)
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            # Step 2: Extract information using AI + embeddings hybrid approach
            logger.info("Extracting information using AI + semantic embeddings...")
            extracted_data = self.extract_with_ai(text)
            
            # Step 3: Validate and clean data
            logger.info("Validating and normalizing data...")
            cleaned_data = self.validate_and_clean_data(extracted_data)
            
            # Step 4: Create Candidate object
            candidate = Candidate(
                name=cleaned_data["name"],
                email=cleaned_data["email"],
                phone=cleaned_data["phone"],
                skills=cleaned_data["skills"],
                experience_years=cleaned_data["experience_years"],
                current_role=cleaned_data["current_role"]
            )
            
            logger.info(f"Fast parsing completed for {candidate.name}")
            logger.info(f"Extracted: {len(candidate.skills)} skills, {candidate.experience_years} years experience")
            
            # Return both candidate and extraction method
            extraction_method = getattr(self, 'extraction_method', 'Unknown')
            return candidate, extraction_method
            
        except Exception as e:
            logger.error(f"Error in resume parsing: {e}")
            raise