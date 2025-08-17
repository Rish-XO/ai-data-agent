from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Dict, Any
import shutil
import os
from pathlib import Path
import logging

from models import Candidate
from ai_langchain_parser import AIResumeParser
from database import db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Resume Parser",
    description="Simple resume parser using LangChain and HuggingFace",
    version="1.0.0"
)

# Enable CORS with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize AI parser
parser = AIResumeParser()

# Mount static files - serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/")
def read_root():
    # Serve the frontend HTML
    frontend_file = frontend_path / "index.html"
    if frontend_file.exists():
        return FileResponse(str(frontend_file))
    return {"message": "AI Resume Parser API is running!"}

@app.post("/upload-resume", response_model=Dict[str, Any])
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse a resume PDF"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file temporarily
        temp_file = UPLOAD_DIR / file.filename
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse resume
        logger.info(f"Parsing resume: {file.filename}")
        candidate = parser.parse_resume(str(temp_file))
        
        # Save to database
        saved_candidate = db.save_candidate(candidate)
        
        # Clean up temp file
        os.remove(temp_file)
        
        return {
            "message": "Resume parsed successfully",
            "candidate": saved_candidate
        }
        
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        # Clean up on error
        if temp_file.exists():
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidates", response_model=List[Dict[str, Any]])
def get_all_candidates():
    """Get all candidates from database"""
    try:
        candidates = db.get_all_candidates()
        return candidates
    except Exception as e:
        logger.error(f"Error fetching candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search", response_model=List[Dict[str, Any]])
def search_candidates(skill: str, semantic: bool = True):
    """Search candidates by skill using semantic similarity"""
    try:
        if not skill:
            raise HTTPException(status_code=400, detail="Skill parameter is required")
        
        # Use semantic search by default
        if semantic:
            candidates = db.search_by_skill_semantic(skill)
            logger.info(f"Semantic search for '{skill}' returned {len(candidates)} results")
        else:
            candidates = db.search_by_skill(skill)
            logger.info(f"Exact search for '{skill}' returned {len(candidates)} results")
        
        return candidates
    except Exception as e:
        logger.error(f"Error searching candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Resume Parser"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Resume Parser API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)