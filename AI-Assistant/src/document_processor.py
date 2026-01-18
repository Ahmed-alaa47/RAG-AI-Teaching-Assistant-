from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS, USE_TOKEN_SPLITTING, ENABLE_OCR
import logging

# Import token-based splitter
try:
    from langchain_text_splitters import TokenTextSplitter
    import tiktoken
    TOKEN_SPLITTER_AVAILABLE = True
except ImportError:
    TOKEN_SPLITTER_AVAILABLE = False
    logging.warning("tiktoken not installed. Using character-based splitting.")

# Import OCR libraries
OCR_AVAILABLE = False
if ENABLE_OCR:
    try:
        import pytesseract
        # from pdf2image import convert_from_path  # Unused, as we use fitz for image extraction
        from PIL import Image
        import fitz  # PyMuPDF
        
        # Set Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        OCR_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("OCR enabled - will extract text from images in PDFs")
    except ImportError as e:
        logging.warning(f"OCR libraries not installed: {e}")
        logging.warning("Install: pip install pytesseract pymupdf Pillow")
        logging.warning("Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Choose splitter based on settings and availability
        if USE_TOKEN_SPLITTING and TOKEN_SPLITTER_AVAILABLE:
            logger.info(f"Using TOKEN-based text splitter (chunk_size={chunk_size} tokens)")
            self.text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            logger.info(f"Using CHARACTER-based text splitter (chunk_size={chunk_size} chars)")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        try:
            if extension == '.pdf':
                # Try OCR if enabled
                if ENABLE_OCR and OCR_AVAILABLE:
                    logger.info(f"Loading PDF with OCR: {path.name}")
                    documents = self._load_pdf_with_ocr(file_path)
                    if documents:
                        return documents
                    logger.info("OCR failed, falling back to standard PDF loader")
                
                # Standard PDF loading
                loader = PyPDFLoader(file_path)
            elif extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif extension == '.pptx':
                loader = UnstructuredPowerPointLoader(file_path)
            elif extension == '.txt':
                loader = TextLoader(file_path)
            else:
                return []
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def _load_pdf_with_ocr(self, file_path: str) -> List[Document]:
        """Load PDF and extract text from images using OCR."""
        try:
            import fitz  # PyMuPDF
            
            documents = []
            pdf_document = fitz.open(file_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract regular text first
                text = page.get_text()
                
                # If page has little text, try OCR on images
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
            logger.info(f"Loaded {len(documents)} pages with OCR from {Path(file_path).name}")
            return documents
        
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        path = Path(directory_path)
        all_documents = []
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs = self.load_document(str(file_path))
                all_documents.extend(docs)
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory_path}")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks
    
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