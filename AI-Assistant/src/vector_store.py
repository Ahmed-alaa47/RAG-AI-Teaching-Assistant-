from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from src.embeddings import EmbeddingManager
from config.settings import CHROMA_DB_DIR, COLLECTION_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, persist_directory: str = CHROMA_DB_DIR, 
                 collection_name: str = COLLECTION_NAME):
        """Initialize vector store manager."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents."""
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        embeddings = self.embedding_manager.get_embeddings()
        
        # Delete existing collection if it exists
        try:
            import shutil
            import os
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info("Removed old vector store")
        except Exception as e:
            logger.warning(f"Could not remove old vector store: {e}")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        # Verify documents were added
        try:
            count = self.vector_store._collection.count()
            logger.info(f"✓ Vector store created with {count} documents and persisted to {self.persist_directory}")
        except:
            logger.info(f"Vector store created and persisted to {self.persist_directory}")
        
        return self.vector_store
    
    def load_vector_store(self) -> Chroma:
        """Load an existing vector store."""
        logger.info("Loading existing vector store")
        embeddings = self.embedding_manager.get_embeddings()
        
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            collection_name=self.collection_name
        )
        
        # Check if vector store has documents
        try:
            count = self.vector_store._collection.count()
            logger.info(f"Vector store loaded with {count} documents")
        except:
            logger.info("Vector store loaded successfully")
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vector_store is None:
            self.load_vector_store()
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.vector_store.add_documents(documents)
        logger.info("Documents added successfully")
    
    def get_vector_store(self) -> Chroma:
        """Get the vector store (load if not already loaded)."""
        if self.vector_store is None:
            self.load_vector_store()
        return self.vector_store