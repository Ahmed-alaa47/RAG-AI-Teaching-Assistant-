from src.rag_pipeline import RAGPipeline
from utils.helpers import format_sources, print_divider
from config.settings import RAW_DATA_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the RAG chatbot."""
    print("=" * 60)
    print("Course Material AI Assistant")
    print("مساعد المواد الدراسية الذكي")
    print("=" * 60)
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Check if vector store exists and has documents
    try:
        logger.info("Attempting to load existing vector store...")
        rag.vector_store_manager.load_vector_store()
        
        # Check if it has documents
        count = rag.vector_store_manager.vector_store._collection.count()
        
        if count == 0:
            raise ValueError("Vector store is empty, need to rebuild")
        
        rag.is_initialized = True
        print(f"\n✓ Loaded existing course materials database ({count} documents)")
    except Exception as e:
        logger.info(f"Need to initialize: {str(e)}")
        print("\n⚙ Processing course materials...")
        print(f"   Reading files from: {RAW_DATA_DIR}")
        
        try:
            rag.initialize()
            
            # Verify it worked
            count = rag.vector_store_manager.vector_store._collection.count()
            print(f"✓ Course materials processed successfully! ({count} documents)")
        except Exception as init_error:
            print(f"\n✗ Error during initialization: {str(init_error)}")
            print("\nPlease ensure:")
            print(f"1. Course files are in: {RAW_DATA_DIR}")
            print("2. Files are in supported formats: PDF, DOCX, PPTX, TXT, Images (PNG, JPG)")
            return
    
    print("\n" + "=" * 60)
    print("You can now ask questions about your course materials or provide a YouTube URL!")
    print("يمكنك الآن طرح أسئلة حول موادك الدراسية أو تزويدنا برابط يوتيوب!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    # Interactive query loop
    history = []
    while True:
        try:
            question = input("\n🎓 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Course Material AI Assistant!")
                print("شكراً لاستخدامك مساعد المواد الدراسية!")
                break
            
            if not question:
                continue
            
            print_divider()
            print("🔍 Searching course materials...\n")
            
            # Get answer with history
            result = rag.query(question, history=history)
            
            # Display answer
            print("📖 Answer:")
            print(result['answer'])
            
            # Update history (keep last 5 interactions to prevent prompt bloat)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result['answer']})
            
            if len(history) > 10:  # 5 user prompts + 5 assistant responses
                history = history[-10:]
            
            # Display sources
            if result['sources']:
                print(format_sources(result['sources']))
            
            print_divider()
        
        except KeyboardInterrupt:
            print("\n\nThank you for using the Course Material AI Assistant!")
            print("شكراً لاستخدامك مساعد المواد الدراسية!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"\n✗ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()