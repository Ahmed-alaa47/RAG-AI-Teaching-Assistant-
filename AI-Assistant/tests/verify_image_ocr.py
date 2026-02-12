import os
import sys
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Add the project root to sys.path to import our modules
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.document_processor import DocumentProcessor
from config.settings import RAW_DATA_DIR

def create_test_image(file_path):
    """Create a simple image with text for OCR testing."""
    # Create a white image
    img = Image.new('RGB', (800, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        # On Windows, Arial is usually available
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Add text
    text = "Artificial Intelligence is the future."
    d.text((10, 10), text, fill=(0, 0, 0), font=font)
    
    img.save(file_path)
    print(f"Test image created at: {file_path}")

def verify_ocr():
    """Verify that DocumentProcessor can load and OCR the image."""
    test_image_path = RAW_DATA_DIR / "ocr_test_image.png"
    create_test_image(test_image_path)
    
    processor = DocumentProcessor()
    print(f"Processing image: {test_image_path}")
    
    documents = processor.load_document(str(test_image_path))
    
    if not documents:
        print("FAILED: No documents loaded from image.")
        return False
    
    content = documents[0].page_content
    print("\n--- Extracted Text ---")
    print(content)
    print("----------------------\n")
    
    # Basic check for keywords
    if "Artificial" in content or "Intelligence" in content:
        print("SUCCESS: OCR extracted English text correctly.")
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        return True
    else:
        print("FAILED: OCR did not extract expected text.")
        return False

if __name__ == "__main__":
    success = verify_ocr()
    sys.exit(0 if success else 1)
