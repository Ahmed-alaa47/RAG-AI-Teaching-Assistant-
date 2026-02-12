import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to sys.path
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline

def verify_timestamps():
    url = "https://www.youtube.com/watch?v=smHa2442Ah4"
    question = f"Look at this video {url} and explain the rule for the dimensions of the output image after a convolution operation. Please mention the time in the video where this is explained."
    
    print("\n" + "="*60)
    print("Verifying YouTube Timestamps Citations")
    print("="*60)
    
    try:
        pipeline = RAGPipeline()
        result = pipeline.query(question)
        
        print("\n--- Answer ---")
        print(result['answer'])
        print("--------------\n")
        
        # Check for [HH:MM:SS] format in answer
        import re
        if re.search(r"\[\d{2}:\d{2}:\d{2}\]", result['answer']):
            print("SUCCESS: Answer contains timestamp citations!")
        else:
            print("WARNING: No timestamp citations found in the answer. Context preview:")
            print(result['context'][:500] + "...")
            
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    verify_timestamps()
