import os
from pathlib import Path

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
COLLECTION_NAME = "course_materials"

# Embedding model settings - Multilingual support
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Supports Arabic & English
EMBEDDING_DEVICE = "cuda"  # Use GPU! Change to "cpu" if you have issues

# Document processing settings
CHUNK_SIZE = 512  # tokens (not characters) - ~350-400 words
CHUNK_OVERLAP = 50  # tokens overlap between chunks
USE_TOKEN_SPLITTING = True  # Use token-based splitting instead of character-based
ENABLE_OCR = True  # Extract text from images in PDFs using OCR

# Retrieval settings
TOP_K_RESULTS = 6  # Increased for better coverage of comparison questions
SIMILARITY_THRESHOLD = 0.2  # Lowered to catch comparison questions

# LLM settings
USE_OLLAMA = True  # Set to False to use simple context display
OLLAMA_MODEL = "llama3.1"  # Best for Arabic & English!
OLLAMA_BASE_URL = "http://localhost:11434"

# Supported file types
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt']