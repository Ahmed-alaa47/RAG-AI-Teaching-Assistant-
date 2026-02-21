import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.rag_pipeline import RAGPipeline
from utils.helpers import fix_arabic_text

def verify():
    print("=" * 60)
    print("Verifying Lecture Identification Fix")
    print("=" * 60)
    
    rag = RAGPipeline()
    try:
        rag.vector_store_manager.load_vector_store()
        rag.is_initialized = True
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    test_queries = [
        "summarize stack lecture",
        "tell me about the trees lecture",
        "linked list lecture"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        result = rag.query(query)
        
        sources = result.get('sources', [])
        print(f"Found {len(sources)} sources")
        
        found_match = False
        for source in sources:
            file_name = source.get('metadata', {}).get('file_name', '')
            content_preview = source.get('content', '')[:100].replace('\n', ' ')
            
            print(f"  - Source: {file_name}")
            print(f"    Content Preview: {content_preview}")
            
            # Check if source content contains the [Source File: ...] header
            if "[Source File:" in source.get('content', ''):
                print("    ✅ Content Header found!")
            else:
                print("    ❌ Content Header MISSING!")

            # Heuristic: does the filename match the topic?
            topic = query.split()[-2 if 'lecture' in query else -1]
            if topic.lower() in file_name.lower():
                found_match = True
                print(f"    ✅ MATCHED topic '{topic}' to filename '{file_name}'")

        if found_match:
            print(f"RESULT: SUCCESS for '{query}'")
        else:
            print(f"RESULT: FAILED for '{query}' - No matching filename found in sources")

    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)

if __name__ == "__main__":
    verify()
