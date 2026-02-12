import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to sys.path
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.youtube_processor import YouTubeProcessor

def debug_youtube():
    url = "https://www.youtube.com/watch?v=smHa2442Ah4&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=4&t=1s"
    processor = YouTubeProcessor()
    
    print("\n" + "="*60)
    print("Debugging YouTube Processor")
    print("="*60)
    
    print(f"Testing URL: {url}")
    
    video_id = processor.extract_video_id(url)
    print(f"Extracted Video ID: {video_id}")
    
    if not video_id:
        print("FAILED: Could not extract Video ID.")
        return
        
    print("\nFetching transcript...")
    transcript = processor.get_transcript(video_id)
    
    if transcript:
        print(f"SUCCESS: Fetched transcript length: {len(transcript)}")
        print("\n--- Transcript Snippet ---")
        print(transcript[:500] + "...")
        print("--------------------------\n")
    else:
        print("FAILED: Could not fetch transcript.")

if __name__ == "__main__":
    debug_youtube()
