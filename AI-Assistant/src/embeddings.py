from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.embeddings = None
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create the embedding model."""
        if self.embeddings is None:
            logger.info(f"Loading embedding model: {self.model_name} on {EMBEDDING_DEVICE}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': EMBEDDING_DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully")
        
        return self.embeddings