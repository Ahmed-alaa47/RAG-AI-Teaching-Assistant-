from typing import List
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager
from config.settings import TOP_K_RESULTS, SIMILARITY_THRESHOLD
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, top_k: int = TOP_K_RESULTS):
        """Initialize the retriever."""
        self.top_k = top_k
        self.vector_store_manager = VectorStoreManager()
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query with similarity filtering."""
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        vector_store = self.vector_store_manager.get_vector_store()
        
        # Get results with scores
        results_with_scores = vector_store.similarity_search_with_score(query, k=self.top_k * 2)
        
        # Filter by similarity threshold
        filtered_results = []
        for doc, score in results_with_scores:
            # Lower score = more similar (distance metric)
            # Convert to similarity: 1 - normalized_distance
            similarity = 1 - min(score, 1.0)
            
            logger.info(f"Document similarity: {similarity:.3f}")
            
            if similarity >= SIMILARITY_THRESHOLD:
                filtered_results.append(doc)
            
            if len(filtered_results) >= self.top_k:
                break
        
        logger.info(f"Retrieved {len(filtered_results)} relevant documents (filtered by threshold)")
        
        return filtered_results
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """Retrieve documents with similarity scores."""
        logger.info(f"Retrieving documents with scores for query: {query[:50]}...")
        
        vector_store = self.vector_store_manager.get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=self.top_k)
        
        logger.info(f"Retrieved {len(results)} documents with scores")
        return results