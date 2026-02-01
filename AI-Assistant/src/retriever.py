from typing import List
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager
from config.settings import TOP_K_RESULTS, SIMILARITY_THRESHOLD
import logging

# logging.basicConfig(level=logging.INFO)  # Removed central logging config
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
        results_with_scores = vector_store.similarity_search_with_score(query, k=self.top_k)
        
        # Filter by similarity threshold if needed
        filtered_results = []
        for doc, score in results_with_scores:
            # ChromaDB returns distance: lower = more similar
            # Distance typically ranges from 0 (identical) to 2 (completely different)
            # Convert to similarity percentage: 0 distance = 100% similarity
            similarity = max(0, 1 - (score / 2))
            
            logger.info(f"Document similarity: {similarity:.3f} (distance: {score:.3f})")
            
            # Apply threshold if > 0
            if SIMILARITY_THRESHOLD > 0:
                if similarity >= SIMILARITY_THRESHOLD:
                    filtered_results.append(doc)
            else:
                # No threshold - return all results
                filtered_results.append(doc)
        
        logger.info(f"Retrieved {len(filtered_results)} relevant documents")
        
        return filtered_results
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """Retrieve documents with similarity scores."""
        logger.info(f"Retrieving documents with scores for query: {query[:50]}...")
        
        vector_store = self.vector_store_manager.get_vector_store()
        results = vector_store.similarity_search_with_score(query, k=self.top_k)
        
        logger.info(f"Retrieved {len(results)} documents with scores")
        return results