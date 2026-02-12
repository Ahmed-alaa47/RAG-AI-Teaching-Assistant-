from typing import Dict
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import Generator
from src.youtube_processor import YouTubeProcessor
from config.settings import RAW_DATA_DIR
import logging
import os

# logging.basicConfig(level=logging.INFO)  # Removed central logging config
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.retriever = Retriever()
        self.generator = Generator()
        self.youtube_processor = YouTubeProcessor()
        self.is_initialized = False
    
    def initialize(self, data_path: str = None):
        """Initialize the pipeline by processing documents and creating vector store."""
        from config.settings import CHUNKS_CACHE_PATH
        if data_path is None:
            data_path = str(RAW_DATA_DIR)
        
        logger.info("Initializing RAG pipeline...")
        
        # Check if cache exists
        chunks = []
        if os.path.exists(CHUNKS_CACHE_PATH):
            logger.info(f"Loading chunks from cache: {CHUNKS_CACHE_PATH}")
            chunks = self.document_processor.load_chunks(CHUNKS_CACHE_PATH)
        
        # If cache empty or not found, process documents
        if not chunks:
            logger.info("Processing documents (no cache found or cache empty)...")
            chunks = self.document_processor.process_documents(data_path)
            
            if not chunks:
                raise ValueError("No documents were processed. Check your data directory.")
            
            # Save to cache
            self.document_processor.save_chunks(chunks, CHUNKS_CACHE_PATH)
        
        # Create vector store
        logger.info("Creating/Updating vector store...")
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
    
    def query(self, question: str, history: list = None) -> Dict:
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
        
        # 1. Fetch YouTube transcript if a URL is provided
        youtube_transcript = self.youtube_processor.process_url(question)
        
        # 2. Standard RAG process (Course Materials)
        try:
            raw_documents = self.retriever.retrieve(question)
            logger.info(f"Retrieved {len(raw_documents)} raw documents")
            
            # Filter out assessment/question-based documents
            documents = []
            assessment_keywords = [
                "which of the following", "question", "solve", 
                "a)", "b)", "c)", "d)", 
                "اختر", "أي مما يلي"
            ]
            
            for doc in raw_documents:
                content_lower = doc.page_content.lower()
                is_assessment = any(keyword in content_lower for keyword in assessment_keywords)
                
                if not is_assessment:
                    documents.append(doc)
                else:
                    logger.info(f"Filtered out assessment document: {doc.page_content[:50]}...")
            
            logger.info(f"Remaining documents after filtering: {len(documents)}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            documents = []

        # 3. Create structured hybrid context
        context_parts = []
        
        if youtube_transcript:
            context_parts.append("[SOURCE: YOUTUBE_VIDEO_TRANSCRIPT]\n" + youtube_transcript)
        
        if documents:
            course_text = "\n\n".join([doc.page_content for doc in documents])
            context_parts.append("[SOURCE: OFFICIAL_COURSE_MATERIALS]\n" + course_text)
        
        if not context_parts:
            return {
                "answer": "I couldn't find any relevant information in the video or course materials.",
                "sources": []
            }
        
        full_context = "\n\n" + "\n\n---\n\n".join(context_parts)
        
        # 4. Generate answer with history
        answer = self.generator.generate_answer(question, full_context, is_youtube=bool(youtube_transcript), history=history)
        
        # Prepare sources for UI
        sources = []
        if youtube_transcript:
            sources.append({"content": "Supplementary Video Transcript", "metadata": {"source": "YouTube"}})
        
        for doc in documents:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context": full_context
        }
    
