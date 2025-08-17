import re
from typing import List
import pypdf
from models import Candidate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common skills to look for in resumes
COMMON_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "React", "Angular", "Vue",
    "Node.js", "Django", "FastAPI", "Flask", "SQL", "PostgreSQL", "MongoDB",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git", "CI/CD", "REST API",
    "Machine Learning", "Data Science", "HTML", "CSS", "C++", "C#", ".NET",
    "Spring", "Express", "Linux", "Windows", "MacOS", "Agile", "Scrum"
]

class SimpleResumeParser:
    """Simple resume parser using only regex and pypdf - no heavy AI models"""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pypdf"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
        return text
    
    def extract_email(self, text: str) -> str:
        """Extract email using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    def extract_phone(self, text: str) -> str:
        """Extract phone number using regex"""
        # Remove all non-numeric except +, (, ), -, space
        phone_patterns = [
            r'\+?1?\s*\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}',  # US format
            r'\+[0-9]{1,3}[\s.-]?[0-9]{4,14}',  # International
            r'\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4}',  # (555) 123-4567
            r'[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',  # 555-123-4567
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
        return None
    
    def extract_name(self, text: str) -> str:
        """Extract name - usually first line or near email"""
        lines = text.split('\n')
        
        # Strategy 1: Look for name near the top of resume
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            # Skip empty lines and lines that look like headers
            if not line or len(line) > 50:
                continue
            # Skip lines with special characters (likely not names)
            if '@' in line or '|' in line or '•' in line:
                continue
            # Skip lines that are all caps (likely section headers)
            if line.isupper():
                continue
            # Check if line contains mostly letters and spaces (likely a name)
            if re.match(r'^[A-Za-z\s\-\.]+$', line):
                # Additional check: should have at least 2 words (first and last name)
                words = line.split()
                if 2 <= len(words) <= 4:  # Most names are 2-4 words
                    return line
        
        # Strategy 2: If no name found, use first non-empty line
        for line in lines:
            line = line.strip()
            if line and len(line) < 50:
                return line
        
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills by matching against common skills list"""
        found_skills = []
        text_lower = text.lower()
        
        for skill in COMMON_SKILLS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience using regex patterns"""
        # Look for patterns like "X years of experience" or "X+ years"
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*',
            r'over\s*(\d+)\s*years?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Fallback: Count years mentioned (like 2018-2023)
        year_pattern = r'(19|20)\d{2}'
        years = re.findall(year_pattern, text)
        if len(years) >= 2:
            try:
                min_year = int(min(years) + years[0][-2:])  # Reconstruct full year
                max_year = int(max(years) + years[0][-2:])
                experience = max_year - min_year
                if 0 < experience < 50:  # Reasonable experience range
                    return experience
            except:
                pass
        
        return 0
    
    def extract_current_role(self, text: str) -> str:
        """Try to extract current job title"""
        # Common job title keywords
        role_keywords = [
            "Software Engineer", "Developer", "Programmer", "Analyst",
            "Manager", "Designer", "Architect", "Consultant", "Lead",
            "Senior", "Junior", "Full Stack", "Frontend", "Backend",
            "DevOps", "Data Scientist", "Product Manager"
        ]
        
        text_lower = text.lower()
        for role in role_keywords:
            if role.lower() in text_lower:
                # Try to get the full title context
                pattern = r'([\w\s]*' + re.escape(role.lower()) + r'[\w\s]*)'
                match = re.search(pattern, text_lower)
                if match:
                    title = match.group(1).strip()
                    # Clean up and capitalize properly
                    title = ' '.join(word.capitalize() for word in title.split())
                    if len(title) < 50:  # Reasonable title length
                        return title
        
        return None
    
    def parse_resume(self, pdf_path: str) -> Candidate:
        """Main method to parse resume and return Candidate object"""
        try:
            # Extract text from PDF
            logger.info(f"Extracting text from {pdf_path}")
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text:
                raise ValueError("No text could be extracted from PDF")
            
            # Extract all information
            name = self.extract_name(text)
            email = self.extract_email(text)
            phone = self.extract_phone(text)
            skills = self.extract_skills(text)
            experience_years = self.extract_experience_years(text)
            current_role = self.extract_current_role(text)
            
            # Create candidate object
            candidate = Candidate(
                name=name,
                email=email,
                phone=phone,
                skills=skills,
                experience_years=experience_years,
                current_role=current_role
            )
            
            logger.info(f"Successfully parsed resume for {candidate.name}")
            logger.info(f"Found {len(skills)} skills, {experience_years} years experience")
            
            return candidate
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            raise