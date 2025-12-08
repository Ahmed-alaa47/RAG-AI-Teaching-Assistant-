import unittest
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.rag_pipeline import RAGPipeline


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()
    
    def test_text_splitter_initialization(self):
        """Test that text splitter is properly initialized."""
        self.assertIsNotNone(self.processor.text_splitter)
        self.assertEqual(self.processor.chunk_size, 512)
    
    def test_supported_extensions(self):
        """Test that correct file extensions are supported."""
        from config.settings import SUPPORTED_EXTENSIONS
        self.assertIn('.pdf', SUPPORTED_EXTENSIONS)
        self.assertIn('.docx', SUPPORTED_EXTENSIONS)
        self.assertIn('.pptx', SUPPORTED_EXTENSIONS)
        self.assertIn('.txt', SUPPORTED_EXTENSIONS)


class TestEmbeddingManager(unittest.TestCase):
    def setUp(self):
        self.embedding_manager = EmbeddingManager()
    
    def test_embedding_initialization(self):
        """Test that embeddings can be initialized."""
        embeddings = self.embedding_manager.get_embeddings()
        self.assertIsNotNone(embeddings)
    
    def test_embedding_generation(self):
        """Test that embeddings can be generated for text."""
        embeddings = self.embedding_manager.get_embeddings()
        test_text = "This is a test sentence."
        result = embeddings.embed_query(test_text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline()
    
    def test_pipeline_initialization(self):
        """Test that pipeline components are initialized."""
        self.assertIsNotNone(self.pipeline.document_processor)
        self.assertIsNotNone(self.pipeline.vector_store_manager)
        self.assertIsNotNone(self.pipeline.retriever)
    
    def test_answer_generation(self):
        """Test answer generation from context."""
        question = "What is machine learning?"
        context = "Machine learning is a subset of artificial intelligence."
        answer = self.pipeline._generate_answer(question, context)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)


if __name__ == '__main__':
    unittest.main()