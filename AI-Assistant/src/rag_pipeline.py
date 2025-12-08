from typing import Dict
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from config.settings import RAW_DATA_DIR, USE_OLLAMA, OLLAMA_MODEL, OLLAMA_BASE_URL
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Install with: pip install ollama")


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.retriever = Retriever()
        self.is_initialized = False
    
    def initialize(self, data_path: str = None):
        """Initialize the pipeline by processing documents and creating vector store."""
        if data_path is None:
            data_path = str(RAW_DATA_DIR)
        
        logger.info("Initializing RAG pipeline...")
        
        # Process documents
        logger.info("Processing documents...")
        chunks = self.document_processor.process_documents(data_path)
        
        if not chunks:
            raise ValueError("No documents were processed. Check your data directory.")
        
        # Create vector store
        logger.info("Creating vector store...")
        self.vector_store_manager.create_vector_store(chunks)
        
        self.is_initialized = True
        logger.info("RAG pipeline initialized successfully!")
    
    def add_documents(self, source_path: str):
        """Add new documents to existing vector store."""
        logger.info(f"Adding documents from {source_path}")
        
        chunks = self.document_processor.process_documents(source_path)
        
        if not chunks:
            logger.warning("No new documents were processed")
            return
        
        self.vector_store_manager.add_documents(chunks)
        logger.info("Documents added successfully!")
    
    def query(self, question: str) -> Dict:
        """Query the RAG system and return answer with sources."""
        if not self.is_initialized:
            logger.info("Pipeline not initialized, loading existing vector store...")
            try:
                self.vector_store_manager.load_vector_store()
                self.is_initialized = True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                return {
                    "answer": "Vector store not found. Please run initialization first.",
                    "sources": []
                }
        
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        try:
            documents = self.retriever.retrieve(question)
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Debug: print first document
            if documents:
                logger.info(f"First document preview: {documents[0].page_content[:100]}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            documents = []
        
        if not documents:
            return {
                "answer": "I couldn't find any relevant information in the course materials.",
                "sources": []
            }
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # Generate answer
        answer = self._generate_answer(question, context)
        
        # Extract source information
        sources = []
        for doc in documents:
            source_info = {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer from context using Ollama or simple extraction."""
        
        # Check if Ollama should be used and is available
        if USE_OLLAMA and OLLAMA_AVAILABLE:
            try:
                return self._generate_with_ollama(question, context)
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                logger.info("Falling back to simple context display")
        
        # Fallback: Simple context display
        return f"Based on the course materials:\n\n{context[:800]}..."
    
    def _generate_with_ollama(self, question: str, context: str) -> str:
        """Generate answer using Ollama local LLM with language detection."""
        
        # Detect if question contains Arabic characters
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
        
        if has_arabic:
            # Arabic prompt
            prompt = f"""بناءً على المحتوى التعليمي التالي، أجب على سؤال الطالب بوضوح وبشكل تعليمي.

المحتوى التعليمي:
{context}

سؤال الطالب: {question}

الإجابة (كن واضحاً وتعليمياً):"""
        else:
            # English prompt
            prompt = f"""Based on the following course material, answer the student's question clearly and concisely.

Course Material:
{context}

Student Question: {question}

Answer (be educational and clear):"""

        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama error: {e}. Make sure Ollama is installed and '{OLLAMA_MODEL}' model is downloaded.")