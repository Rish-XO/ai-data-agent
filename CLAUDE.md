# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Note
This is a **personal learning project** designed to understand and practice with modern AI tools like LangChain, HuggingFace, and FastAPI. The code prioritizes simplicity and learning over production-ready features. Keep implementations simple and focused on core functionality.

## Development Commands

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start FastAPI server (from backend directory)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Project Architecture

This is an AI-powered data agent application with the following structure:

- **backend/**: FastAPI application with LangChain integration
  - Uses LangChain (0.0.350) for AI orchestration
  - FastAPI (0.104.1) for REST API
  - PyPDF2 for document processing
  - Transformers and Torch for ML models
  - PostgreSQL support via psycopg2

- **frontend/**: Frontend application (currently empty)
- **docs/**: Documentation directory
- **resumes/**: Directory for resume/document storage

## Key Technologies

- **LangChain**: Used for building AI chains and agents
- **Hugging Face Transformers**: For embedding and language models
- **PostgreSQL**: Database backend
- **python-dotenv**: Environment variable management

## Environment Configuration

The backend uses a `.env` file for configuration. Ensure environment variables are properly configured before running the application.