import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.youtube_processor import YouTubeProcessor

def test_url_extraction():
    processor = YouTubeProcessor()
    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1s"
    ]
    
    expected_id = "dQw4w9WgXcQ"
    
    for url in urls:
        extracted_id = processor.extract_video_id(url)
        print(f"URL: {url} -> Extracted ID: {extracted_id}")
        if extracted_id != expected_id:
            print(f"FAILED: Expected {expected_id}, got {extracted_id}")
            return False
    
    print("SUCCESS: All YouTube URL formats parsed correctly.")
    return True

if __name__ == "__main__":
    success = test_url_extraction()
    sys.exit(0 if success else 1)
