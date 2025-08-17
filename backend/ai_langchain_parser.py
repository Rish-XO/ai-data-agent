import json
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from transformers import pipeline
from models import Candidate
from config import HUGGINGFACE_API_TOKEN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIResumeParser:
    """AI-powered resume parser using LangChain and HuggingFace"""
    
    def __init__(self):
        # Initialize HuggingFace text generation pipeline
        # Using a smaller, efficient model for text generation
        try:
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                max_length=512,
                temperature=0.3,  # Low temperature for consistent output
                do_sample=True,
                pad_token_id=50256
            )
            logger.info("AI model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load AI model: {e}. Using fallback extraction.")
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
            for doc in documents:
                full_text += doc.page_content + "\n"
            
            logger.info(f"Loaded PDF with {len(documents)} pages")
            return full_text
            
        except Exception as e:
            logger.error(f"Error loading PDF with LangChain: {e}")
            raise
    
    def chunk_text_with_langchain(self, text: str) -> List[str]:
        """Split text into chunks using LangChain text splitter"""
        docs = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]
    
    def extract_with_ai(self, text: str) -> Dict[str, Any]:
        """Extract information using AI model"""
        if not self.text_generator:
            return self.fallback_extraction(text)
        
        try:
            # Limit text size for model processing
            text_sample = text[:1500]  # First 1500 chars usually contain key info
            
            # Create prompt
            prompt = self.extraction_prompt.format(resume_text=text_sample)
            
            # Generate response
            logger.info("Generating AI response...")
            response = self.text_generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                pad_token_id=50256
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Find JSON in the response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                extracted_data = json.loads(json_str)
                logger.info("AI extraction successful")
                return extracted_data
            else:
                logger.warning("No valid JSON found in AI response, using fallback")
                return self.fallback_extraction(text)
                
        except Exception as e:
            logger.error(f"AI extraction failed: {e}, using fallback")
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
        
        # Skills extraction
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
        
        # Experience extraction
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
        ]
        experience_years = 0
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                experience_years = int(match.group(1))
                break
        
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
    
    def parse_resume(self, pdf_path: str) -> Candidate:
        """Main method to parse resume using AI and return Candidate object"""
        try:
            logger.info(f"Starting AI-powered resume parsing for: {pdf_path}")
            
            # Step 1: Load PDF using LangChain
            text = self.load_pdf_with_langchain(pdf_path)
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            # Step 2: Extract information using AI
            extracted_data = self.extract_with_ai(text)
            
            # Step 3: Validate and clean data
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
            
            logger.info(f"AI parsing completed for {candidate.name}")
            logger.info(f"Extracted: {len(candidate.skills)} skills, {candidate.experience_years} years experience")
            
            return candidate
            
        except Exception as e:
            logger.error(f"Error in AI resume parsing: {e}")
            raise