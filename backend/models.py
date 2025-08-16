from pydantic import BaseModel, EmailStr
from typing import List, Optional

class Candidate(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    skills: List[str] = []
    experience_years: int = 0
    current_role: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1234567890",
                "skills": ["Python", "FastAPI", "PostgreSQL"],
                "experience_years": 5,
                "current_role": "Software Engineer"
            }
        }