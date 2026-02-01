from typing import Dict
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import Generator
from config.settings import RAW_DATA_DIR
import logging

# logging.basicConfig(level=logging.INFO)  # Removed central logging config
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.retriever = Retriever()
        self.generator = Generator()
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
        answer = self.generator.generate_answer(question, context, question_type=None)
        
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
    
