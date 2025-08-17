import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from config import DATABASE_URL
from models import Candidate
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.connection_string = DATABASE_URL
        self.init_database()
    
    def get_connection(self):
        """Create a database connection"""
        return psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
    
    def init_database(self):
        """Initialize database table if not exists"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS candidates (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            email VARCHAR(255) UNIQUE,
                            phone VARCHAR(50),
                            skills TEXT,
                            experience_years INTEGER DEFAULT 0,
                            "current_role" VARCHAR(255),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def save_candidate(self, candidate: Candidate) -> Dict[str, Any]:
        """Save candidate to database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Convert skills list to JSON string
                    skills_json = json.dumps(candidate.skills)
                    
                    # Insert or update on conflict (email)
                    cursor.execute("""
                        INSERT INTO candidates (name, email, phone, skills, experience_years, "current_role")
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (email) 
                        DO UPDATE SET 
                            name = EXCLUDED.name,
                            phone = EXCLUDED.phone,
                            skills = EXCLUDED.skills,
                            experience_years = EXCLUDED.experience_years,
                            "current_role" = EXCLUDED."current_role"
                        RETURNING *
                    """, (
                        candidate.name,
                        candidate.email,
                        candidate.phone,
                        skills_json,
                        candidate.experience_years,
                        candidate.current_role
                    ))
                    
                    result = cursor.fetchone()
                    conn.commit()
                    
                    # Convert skills back to list
                    if result and result.get('skills'):
                        result['skills'] = json.loads(result['skills'])
                    
                    logger.info(f"Candidate {candidate.name} saved successfully")
                    return dict(result) if result else None
                    
        except Exception as e:
            logger.error(f"Error saving candidate: {e}")
            raise
    
    def get_all_candidates(self) -> List[Dict[str, Any]]:
        """Get all candidates from database"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
                    results = cursor.fetchall()
                    
                    # Convert skills JSON to list for each candidate
                    candidates = []
                    for row in results:
                        candidate = dict(row)
                        if candidate.get('skills'):
                            candidate['skills'] = json.loads(candidate['skills'])
                        candidates.append(candidate)
                    
                    return candidates
                    
        except Exception as e:
            logger.error(f"Error fetching candidates: {e}")
            return []
    
    def search_by_skill(self, skill: str) -> List[Dict[str, Any]]:
        """Search candidates by skill"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Use ILIKE for case-insensitive search in JSON
                    cursor.execute("""
                        SELECT * FROM candidates 
                        WHERE skills ILIKE %s
                        ORDER BY experience_years DESC
                    """, (f'%{skill}%',))
                    
                    results = cursor.fetchall()
                    
                    # Convert skills JSON to list
                    candidates = []
                    for row in results:
                        candidate = dict(row)
                        if candidate.get('skills'):
                            candidate['skills'] = json.loads(candidate['skills'])
                        candidates.append(candidate)
                    
                    logger.info(f"Found {len(candidates)} candidates with skill: {skill}")
                    return candidates
                    
        except Exception as e:
            logger.error(f"Error searching candidates: {e}")
            return []

# Create a singleton instance
db = Database()