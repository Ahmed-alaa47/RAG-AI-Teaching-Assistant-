import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add project root to sys.path
project_root = Path(r"d:\AHMED ALAA\Fourth year\Graduation Project\Graduation\RAG-AI-Teaching-Assistant-\AI-Assistant")
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline

def verify_conversation_history():
    print("\n" + "="*60)
    print("Verifying Conversation History Support")
    print("="*60)
    
    pipeline = RAGPipeline()
    history = []
    
    # Turn 1: Specific question
    q1 = "What is padding in CNNs?"
    print(f"\nUser: {q1}")
    result1 = pipeline.query(q1, history=history)
    print(f"Assistant: {result1['answer']}")
    
    history.append({"role": "user", "content": q1})
    history.append({"role": "assistant", "content": result1['answer']})
    
    # Turn 2: Follow-up question referring to previous turn
    q2 = "Why do we use it?"
    print(f"\nUser: {q2}")
    result2 = pipeline.query(q2, history=history)
    print(f"Assistant: {result2['answer']}")
    
    # Check if AI understood "it" as padding
    keywords = ["padding", "preserve", "size", "dimension", "حشو", "أبعاد"]
    found_context = any(word.lower() in result2['answer'].lower() for word in keywords)
    
    if found_context:
        print("\nSUCCESS: AI correctly used context from history for the follow-up question!")
    else:
        print("\nNOTE: AI response might not have explicitly mentioned the subject, check for continuity.")

if __name__ == "__main__":
    verify_conversation_history()
