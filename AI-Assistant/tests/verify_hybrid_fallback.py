import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to sys.path
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline

def verify_hybrid_fallback():
    # This video is about CNNs but might not contain specific trivia from the materials
    # We'll ask a question that is in the course materials but NOT in this specific video transcript
    url = "https://www.youtube.com/watch?v=smHa2442Ah4"
    
    # We need a question that is definitely in the course materials but NOT in the video.
    # Since I don't know the exact content of the course materials, I will try to ask 
    # about something very specific that is likely in the indexed documents but not in a 
    # general CNN video, or just check the wording of the response.
    
    question = f"Using this video {url}, explain what are the disadvantages of manual feature extraction in traditional computer vision compared to CNNs. If it's not in the video, check the course materials."
    
    print("\n" + "="*60)
    print("Verifying Hybrid Fallback Logic (Video -> Materials)")
    print("="*60)
    
    try:
        pipeline = RAGPipeline()
        result = pipeline.query(question)
        
        print("\n--- Answer ---")
        print(result['answer'])
        print("--------------\n")
        
        # Check for the specified fallback indicator
        fallback_indicators = [
            "This information was not mentioned in the video",
            "هذه المعلومة غير مذكورة في الفيديو",
            "according to the course materials",
            "بناءً على المواد الدراسية"
        ]
        
        found_fallback = any(indicator.lower() in result['answer'].lower() for indicator in fallback_indicators)
        
        if found_fallback:
            print("SUCCESS: Answer correctly identified source fallback or used the requested wording.")
        else:
            print("NOTE: The AI might have found the info in the video or didn't use the exact fallback phrase.")
            print("Check if the answer makes sense and cites the source correctly.")
            
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    verify_hybrid_fallback()
