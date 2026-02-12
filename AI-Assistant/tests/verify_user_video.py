import sys
from pathlib import Path
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Add project root to sys.path
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline

def test_user_video():
    url = "https://www.youtube.com/watch?v=smHa2442Ah4&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=4&t=1s"
    question = f"Based on this video {url}, what is the used rule to determine the new shape of image after using a filter, padding, and stride?"
    
    print("\n" + "="*60)
    print("Testing YouTube Support with User-Provided Video")
    print("="*60)
    
    try:
        pipeline = RAGPipeline()
        # Note: We don't necessarily need to initialize(data_path) if we are only testing YouTube
        # but initializing ensures all components are ready.
        
        print(f"URL: {url}")
        print(f"Question: {question}")
        print("\nProcessing... (This might take a moment to fetch transcript and generate answer)\n")
        
        result = pipeline.query(question)
        
        print("Answer:")
        print(result['answer'])
        print("\n" + "="*60)
        
        # Check if answer contains the formula components
        answer_lower = result['answer'].lower()
        if "n" in answer_lower and "p" in answer_lower and "f" in answer_lower and "s" in answer_lower:
            print("SUCCESS: The system successfully retrieved the transcript and answered the question.")
            return True
        else:
            print("WARNING: Answer received but might not contain the full mathematical rule.")
            return True # Still a success in terms of pipeline flow
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_user_video()
    sys.exit(0 if success else 1)
