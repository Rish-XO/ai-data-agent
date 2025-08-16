import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from models import Candidate
from config import HUGGINGFACE_API_TOKEN
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common skills to look for in resumes
COMMON_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "React", "Angular", "Vue",
    "Node.js", "Django", "FastAPI", "Flask", "SQL", "PostgreSQL", "MongoDB",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "CI/CD", "REST API",
    "Machine Learning", "Data Science", "LangChain", "AI", "NLP"
]

class ResumeParser:
    def __init__(self):
        # Initialize HuggingFace pipeline for NER (Named Entity Recognition)
        # Using a small, free model
        try:
            self.ner_pipeline = pipeline(
                "ner", 
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.warning(f"Could not load NER model: {e}. Using regex fallback only.")
            self.ner_pipeline = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using LangChain"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = " ".join([page.page_content for page in pages])
        return text
    
    def extract_email(self, text: str) -> str:
        """Extract email using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    def extract_phone(self, text: str) -> str:
        """Extract phone number using regex"""
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\d{1,14}',  # International format
            r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # 123-456-7890
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def extract_name(self, text: str) -> str:
        """Extract name using NER or fallback to first line"""
        if self.ner_pipeline:
            try:
                # Use first 200 characters for name extraction
                entities = self.ner_pipeline(text[:200])
                for entity in entities:
                    if entity['entity_group'] == 'PER':  # Person entity
                        return entity['word']
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        # Fallback: assume name is in first non-empty line
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 50:  # Reasonable name length
                return line
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills by matching against common skills list"""
        found_skills = []
        text_lower = text.lower()
        
        for skill in COMMON_SKILLS:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience using regex patterns"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Count year mentions as rough estimate
        year_mentions = re.findall(r'20\d{2}|19\d{2}', text)
        if len(year_mentions) >= 2:
            return 2024 - int(min(year_mentions))  # Rough estimate
        
        return 0
    
    def parse_resume(self, pdf_path: str) -> Candidate:
        """Main method to parse resume and return Candidate object"""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Extract information
            candidate = Candidate(
                name=self.extract_name(text),
                email=self.extract_email(text),
                phone=self.extract_phone(text),
                skills=self.extract_skills(text),
                experience_years=self.extract_experience_years(text),
                current_role=None  # Could be enhanced later
            )
            
            logger.info(f"Successfully parsed resume for {candidate.name}")
            return candidate
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise