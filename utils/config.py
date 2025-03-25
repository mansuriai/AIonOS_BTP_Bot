# utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project structure
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DB_DIR = BASE_DIR / "storage" / "vectordb"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0" 
    EMBEDDING_DIMENSION = 1024  # Adjust based on your specific embedding model
    LLM_MODEL = "gpt-4o-mini"
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT_TOKENS = 4000

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Pinecone settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    PINECONE_INDEX_NAME = "aionos-assistant"
    
    # App settings
    APP_TITLE = "AIonOSAssist"
    MAX_HISTORY_LENGTH = 8
    
    # Vector DB settings
    COLLECTION_NAME = "aionos-documents"
    DISTANCE_METRIC = "cosine"
    
    # Pinecone index settings
    PINECONE_INDEX_SPEC = {
        "cloud": "aws",
        "region": "us-east-1",
        "metric": "cosine"
    }

    # PDF_STORAGE_DIR = BASE_DIR / "pdfs"
    
config = Config()