"""
Embeddings Manager for Semantic Search
Handles skill embeddings and similarity matching using sentence-transformers
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expanded skill list with common variations
SKILL_VARIATIONS = {
    "JavaScript": ["JavaScript", "JS", "Javascript", "java script"],
    "TypeScript": ["TypeScript", "TS", "Typescript", "type script"],
    "Python": ["Python", "Python3", "Python2", "Py"],
    "React": ["React", "ReactJS", "React.js", "React JS"],
    "Angular": ["Angular", "AngularJS", "Angular.js"],
    "Vue": ["Vue", "VueJS", "Vue.js", "Vue JS"],
    "Node.js": ["Node.js", "NodeJS", "Node", "Node JS"],
    "Java": ["Java", "JVM", "Java SE", "Java EE"],
    "C++": ["C++", "CPP", "C Plus Plus"],
    "C#": ["C#", "CSharp", "C Sharp", ".NET"],
    "SQL": ["SQL", "MySQL", "PostgreSQL", "SQLite", "Database"],
    "MongoDB": ["MongoDB", "Mongo", "NoSQL"],
    "Docker": ["Docker", "Containerization", "Containers"],
    "Kubernetes": ["Kubernetes", "K8s", "K8S"],
    "AWS": ["AWS", "Amazon Web Services", "EC2", "S3", "Lambda"],
    "Azure": ["Azure", "Microsoft Azure", "Azure Cloud"],
    "Git": ["Git", "GitHub", "GitLab", "Version Control"],
    "Machine Learning": ["Machine Learning", "ML", "Deep Learning", "AI", "Neural Networks"],
    "Data Science": ["Data Science", "Data Analysis", "Analytics", "Data Mining"],
    "Django": ["Django", "Django REST", "DRF"],
    "Flask": ["Flask", "Flask API"],
    "FastAPI": ["FastAPI", "Fast API"],
    "Spring": ["Spring", "Spring Boot", "Spring Framework"],
    "Express": ["Express", "ExpressJS", "Express.js"],
    "REST API": ["REST API", "RESTful", "REST", "Web API"],
    "GraphQL": ["GraphQL", "Graph QL"],
    "Linux": ["Linux", "Ubuntu", "CentOS", "Unix"],
    "DevOps": ["DevOps", "CI/CD", "Jenkins", "GitOps"],
    "Agile": ["Agile", "Scrum", "Kanban", "Sprint"],
}

class EmbeddingsManager:
    """Manages skill embeddings and semantic similarity"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a pre-trained sentence transformer model
        all-MiniLM-L6-v2: Good balance of speed and quality (80MB)
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
            
            # Pre-compute embeddings for known skills
            self.skill_embeddings = self._precompute_skill_embeddings()
            logger.info(f"Pre-computed embeddings for {len(self.skill_embeddings)} skills")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None
            self.skill_embeddings = {}
    
    def _precompute_skill_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all known skills"""
        embeddings = {}
        
        for canonical_skill, variations in SKILL_VARIATIONS.items():
            # Create embedding for canonical skill name
            canonical_embedding = self.model.encode(canonical_skill)
            embeddings[canonical_skill] = canonical_embedding
            
            # Also store embeddings for variations for faster lookup
            for variation in variations:
                variation_embedding = self.model.encode(variation)
                embeddings[variation] = variation_embedding
        
        return embeddings
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for any text"""
        if not self.model:
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def find_similar_skills(self, skill_text: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find similar skills from known skills using cosine similarity
        Returns list of (skill_name, similarity_score) tuples
        """
        if not self.model or not self.skill_embeddings:
            return []
        
        try:
            # Create embedding for input skill
            skill_embedding = self.create_embedding(skill_text)
            
            # Calculate similarity with all known skills
            similarities = []
            checked_canonical = set()
            
            for canonical_skill, variations in SKILL_VARIATIONS.items():
                if canonical_skill in checked_canonical:
                    continue
                    
                # Get canonical skill embedding
                canonical_embedding = self.skill_embeddings.get(canonical_skill)
                if canonical_embedding is None:
                    continue
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    skill_embedding.reshape(1, -1),
                    canonical_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= threshold:
                    similarities.append((canonical_skill, float(similarity)))
                    checked_canonical.add(canonical_skill)
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
            
        except Exception as e:
            logger.error(f"Error finding similar skills: {e}")
            return []
    
    def extract_skills_semantic(self, text: str, threshold: float = 0.7) -> List[str]:
        """
        Extract skills from text using semantic similarity
        Returns list of canonical skill names
        """
        if not self.model:
            return []
        
        extracted_skills = set()
        
        # Split text into potential skill phrases (1-3 words)
        words = text.split()
        potential_skills = []
        
        # Single words
        potential_skills.extend(words)
        
        # Two-word phrases
        for i in range(len(words) - 1):
            potential_skills.append(f"{words[i]} {words[i+1]}")
        
        # Three-word phrases
        for i in range(len(words) - 2):
            potential_skills.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Check each potential skill
        for potential in potential_skills:
            # Skip very short or very long strings
            if len(potential) < 2 or len(potential) > 30:
                continue
            
            # Find similar known skills
            similar_skills = self.find_similar_skills(potential, threshold)
            
            for skill, score in similar_skills:
                if score >= threshold:
                    extracted_skills.add(skill)
                    logger.debug(f"Matched '{potential}' to '{skill}' (score: {score:.2f})")
        
        return list(extracted_skills)
    
    def search_candidates_semantic(self, query: str, candidate_skills_list: List[List[str]], 
                                  top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search candidates using semantic similarity
        Returns list of (candidate_index, similarity_score) tuples
        """
        if not self.model:
            return []
        
        try:
            # Create embedding for search query
            query_embedding = self.create_embedding(query)
            
            # Find similar skills to the query
            query_skills = self.find_similar_skills(query, threshold=0.6)
            
            # Score each candidate
            candidate_scores = []
            
            for idx, candidate_skills in enumerate(candidate_skills_list):
                if not candidate_skills:
                    continue
                
                # Create combined embedding for candidate skills
                skill_embeddings = []
                for skill in candidate_skills:
                    if skill in self.skill_embeddings:
                        skill_embeddings.append(self.skill_embeddings[skill])
                
                if not skill_embeddings:
                    continue
                
                # Average the skill embeddings
                avg_embedding = np.mean(skill_embeddings, axis=0)
                
                # Calculate similarity with query
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    avg_embedding.reshape(1, -1)
                )[0][0]
                
                candidate_scores.append((idx, float(similarity)))
            
            # Sort by score and return top k
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            return candidate_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Normalize skill names to canonical forms using embeddings
        E.g., "ReactJS" → "React", "Python3" → "Python"
        """
        normalized = set()
        
        for skill in skills:
            # Find most similar canonical skill
            similar = self.find_similar_skills(skill, threshold=0.85)
            
            if similar:
                # Use the most similar canonical skill
                normalized.add(similar[0][0])
            else:
                # Keep original if no match found
                normalized.add(skill)
        
        return list(normalized)
    
    def embedding_to_json(self, embedding: np.ndarray) -> str:
        """Convert numpy embedding to JSON string for storage"""
        if embedding is None:
            return "[]"
        return json.dumps(embedding.tolist())
    
    def json_to_embedding(self, json_str: str) -> np.ndarray:
        """Convert JSON string back to numpy embedding"""
        if not json_str or json_str == "[]":
            return np.zeros(self.embedding_dim)
        return np.array(json.loads(json_str))

# Create singleton instance
embeddings_manager = EmbeddingsManager()