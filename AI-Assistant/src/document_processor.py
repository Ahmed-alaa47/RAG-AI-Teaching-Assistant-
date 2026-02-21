from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable
import logging
import os
import pickle

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)

from config.settings import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    ENABLE_OCR, 
    TESSERACT_CMD
)

# Configure logging
logger = logging.getLogger(__name__)

# OCR Dependencies and Configuration
OCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    
    # Configure Tesseract from settings
    if TESSERACT_CMD:
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        else:
            # If explicit path is provided but missing, warn user. 
            # If it's just 'tesseract', shutil.which or system lookup applies.
            if os.path.isabs(TESSERACT_CMD) and not os.path.exists(TESSERACT_CMD):
                 logger.warning(f"Configured Tesseract executable not found at: {TESSERACT_CMD}")
            else:
                 pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    OCR_AVAILABLE = True
    logger.info("OCR libraries available. Image text extraction enabled.")

except ImportError as e:
    logger.warning(f"OCR libraries not installed: {e}")
    logger.warning("To enable OCR (PDF images/scans + Image files), install: pip install pytesseract pymupdf Pillow")


class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Using CHARACTER-based text splitter (chunk_size={chunk_size} chars)")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # Loader Registry
        self.loader_mapping: Dict[str, Callable[[str], List[Document]]] = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.pptx': self._load_pptx,
            '.txt': self._load_txt,
            '.png': self._load_image,
            '.jpg': self._load_image,
            '.jpeg': self._load_image,
        }
    
    def process_documents(self, source_path: str) -> List[Document]:
        """Complete pipeline: load and split documents."""
        path = Path(source_path)
        
        if path.is_file():
            documents = self.load_document(str(path))
        elif path.is_dir():
            documents = self.load_directory(str(path))
        else:
            logger.error(f"Invalid path: {source_path}")
            return []
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        chunks = self.split_documents(documents)
        return chunks

    def save_chunks(self, chunks: List[Document], cache_path: str):
        """Save chunks to a pickle file."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"Saved {len(chunks)} chunks to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving chunks to cache: {e}")

    def load_chunks(self, cache_path: str) -> List[Document]:
        """Load chunks from a pickle file."""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    chunks = pickle.load(f)
                logger.info(f"Loaded {len(chunks)} chunks from cache: {cache_path}")
                return chunks
        except Exception as e:
            logger.error(f"Error loading chunks from cache: {e}")
        return []

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        path = Path(directory_path)
        all_documents = []
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.loader_mapping:
                docs = self.load_document(str(file_path))
                all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents

    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.loader_mapping:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        loader_func = self.loader_mapping.get(extension)
        if not loader_func:
            logger.warning(f"No loader registered for extension: {extension}")
            return []

        try:
            # Execute specific loader
            documents = loader_func(file_path)
            
            # Enrich metadata
            self._enrich_metadata(documents, path)
            
            logger.info(f"Loaded {len(documents)} pages/items from {path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks and prepend filename for context."""
        chunks = self.text_splitter.split_documents(documents)
        
        # Prepend filename to each chunk's content for better context and retrieval
        for chunk in chunks:
            file_name = chunk.metadata.get("file_name", "Unknown Source")
            chunk.page_content = f"[Source File: {file_name}]\n{chunk.page_content}"
            
        logger.info(f"Split into {len(chunks)} chunks with filename headers")
        return chunks

    def _enrich_metadata(self, documents: List[Document], path: Path):
        """Add standard metadata to all documents."""
        for doc in documents:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_type"] = path.suffix.lower()
            doc.metadata["ingestion_timestamp"] = datetime.now().isoformat()
            # Ensure source is consistent
            doc.metadata["source"] = str(path.absolute())

    # --- Specific Loaders ---

    def _load_docx(self, file_path: str) -> List[Document]:
        return Docx2txtLoader(file_path).load()

    def _load_pptx(self, file_path: str) -> List[Document]:
        return UnstructuredPowerPointLoader(file_path).load()

    def _load_txt(self, file_path: str) -> List[Document]:
        return TextLoader(file_path).load()

    def _load_image(self, file_path: str) -> List[Document]:
        """Load standalone image file and extract text using OCR."""
        if not (OCR_AVAILABLE and ENABLE_OCR):
            logger.warning(f"OCR not available or disabled. Cannot process image: {file_path}")
            return []

        try:
            image = Image.open(file_path)
            # Extract text (Arabic + English)
            ocr_text = pytesseract.image_to_string(image, lang='eng+ara')
            
            if ocr_text.strip():
                return [Document(page_content=ocr_text, metadata={"source": file_path})]
            else:
                logger.info(f"No text extracted from image: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return []

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF with fallback to OCR if needed."""
        # Hybrid approach: Try OCR first if enabled, checking internal logic,
        # otherwise standard load.
        
        path = Path(file_path)
        
        # 1. Try OCR logic if enabled
        if ENABLE_OCR and OCR_AVAILABLE:
            logger.info(f"Attempting PDF load with OCR check: {path.name}")
            documents = self._load_pdf_with_ocr(file_path)
            if documents:
                return documents
            logger.info("OCR extraction yielded no results or failed, falling back to standard loader.")

        # 2. Standard loading
        return PyPDFLoader(file_path).load()

    def _load_pdf_with_ocr(self, file_path: str) -> List[Document]:
        """Load PDF and extract text from images using OCR if text is sparse."""
        # Preserving original logic and heuristics exactly as requested.
        try:
            documents = []
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract regular text first
                text = page.get_text()
                
                # If page has little text, try OCR on images (Heuristic: < 100 chars)
                if len(text.strip()) < 100:
                    # Get images from page
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Convert to PIL Image
                            from io import BytesIO
                            image = Image.open(BytesIO(image_bytes))
                            
                            # Extract text using OCR (Arabic + English)
                            ocr_text = pytesseract.image_to_string(image, lang='eng+ara')
                            
                            if ocr_text.strip():
                                text += f"\n[OCR from image {img_index + 1}]\n{ocr_text}"
                                logger.info(f"Extracted {len(ocr_text)} characters via OCR from page {page_num + 1}")
                        
                        except Exception as img_error:
                            logger.warning(f"Could not process image on page {page_num + 1}: {img_error}")
                
                # Create document for this page
                if text.strip():
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1
                        }
                    )
                    documents.append(doc)
            
            pdf_document.close()
            # If no documents extracted via this method (and we didn't crash), 
            # we return what we found. If list is empty, caller falls back.
            # (Note: Original code returned documents if any were found, else fell back?)
            # Original code: 
            #   documents = self._load_pdf_with_ocr(file_path)
            #   if documents: return documents
            # So if this returns [] or fails, we fall back.
            
            if documents:
                logger.info(f"Loaded {len(documents)} pages via custom PDF logic from {Path(file_path).name}")
            
            return documents
        
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return []