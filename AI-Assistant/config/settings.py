import os
from pathlib import Path
from dotenv import load_dotenv

#Load environment variables from .env file
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Vector store settings
CHROMA_DB_DIR = str(PROCESSED_DATA_DIR / "chroma_db")
CHUNKS_CACHE_PATH = PROCESSED_DATA_DIR / "processed_chunks.pkl"
COLLECTION_NAME = "course_materials"

# Embedding model settings - Multilingual support
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Supports Arabic & English
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # Use CPU by default in Docker, or 'cuda' if GPU is available

# Document processing settings
CHUNK_SIZE = 800  # characters (approx 150-250 words)
CHUNK_OVERLAP = 150  # characters overlap
ENABLE_OCR = True  # Extract text from images in PDFs using OCR

# Retrieval settings
TOP_K_RESULTS = 4  # Increased for better context coverage
SIMILARITY_THRESHOLD = 0.0  # No threshold - retrieve all TOP_K results

# LLM settings - Groq API
USE_GROQ = True  # Changed from USE_OLLAMA
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Llama 3.3 70B model

# Use host.docker.internal as default if running inside Docker and no URL is provided
default_url = "http://host.docker.internal:11434" if os.path.exists('/.dockerenv') else "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", default_url)

# OCR Settings
# Allow overriding via environment variable, otherwise try default paths
TESSERACT_CMD = os.getenv("TESSERACT_PATH", r'C:\Program Files\Tesseract-OCR\tesseract.exe' if os.name == 'nt' else '/usr/bin/tesseract')

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt', '.png', '.jpg', '.jpeg']