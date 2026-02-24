from typing import Dict
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import Generator
from src.youtube_processor import YouTubeProcessor
from src.recommender import RecommendationEngine
from src.presentation_maker import PresentationMaker
from config.settings import RAW_DATA_DIR
import logging
import os
import re

# logging.basicConfig(level=logging.INFO)  # Removed central logging config
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.retriever = Retriever()
        self.generator = Generator()
        self.youtube_processor = YouTubeProcessor()
        self.recommender = RecommendationEngine()
        self.presentation_maker = PresentationMaker()
        self.is_initialized = False
    
    def initialize(self, data_path: str = None):
        """Initialize the pipeline by processing documents and creating vector store."""
        from config.settings import CHUNKS_CACHE_PATH
        if data_path is None:
            data_path = str(RAW_DATA_DIR)
        
        logger.info("Initializing RAG pipeline...")
        
        # Check if cache exists
        chunks = []
        if os.path.exists(CHUNKS_CACHE_PATH):
            logger.info(f"Loading chunks from cache: {CHUNKS_CACHE_PATH}")
            chunks = self.document_processor.load_chunks(CHUNKS_CACHE_PATH)
        
        # If cache empty or not found, process documents
        if not chunks:
            logger.info("Processing documents (no cache found or cache empty)...")
            chunks = self.document_processor.process_documents(data_path)
            
            if not chunks:
                raise ValueError("No documents were processed. Check your data directory.")
            
            # Save to cache
            self.document_processor.save_chunks(chunks, CHUNKS_CACHE_PATH)
        
        # Create vector store
        logger.info("Creating/Updating vector store...")
        self.vector_store_manager.create_vector_store(chunks)
        
        self.is_initialized = True
        logger.info("RAG pipeline initialized successfully!")
    
    def add_documents(self, source_path: str):
        """Add new documents to existing vector store."""
        logger.info(f"Adding documents from {source_path}")
        
        chunks = self.document_processor.process_documents(source_path)
        
        if not chunks:
            logger.warning("No new documents were processed")
            return
        
        self.vector_store_manager.add_documents(chunks)
        logger.info("Documents added successfully!")
    
    def query(self, question: str, history: list = None) -> Dict:
        """Query the RAG system and return answer with sources."""
        if not self.is_initialized:
            logger.info("Pipeline not initialized, loading existing vector store...")
            try:
                self.vector_store_manager.load_vector_store()
                self.is_initialized = True
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                return {
                    "answer": "Vector store not found. Please run initialization first.",
                    "sources": []
                }
        
        logger.info(f"Processing query: {question}")
        
        # 0. Intent Detection & Query Cleaning
        question_lower = question.lower()
        presentation_keywords = [
            "presentation", "slides", "powerpoint", "pptx", "make a presentation",
            "عرض تقديمي", "شرائح", "بوربوينت", "اعمل عرض", "سوي بريزنتيشن"
        ]
        recommendation_keywords = [
            "recommend", "suggest", "more resources", "other courses", "another video",
            "مقترح", "ترشيح", "مصادر أخرى", "كورس آخر", "نرشح", "زيدني"
        ]
        
        is_presentation = any(keyword in question_lower for keyword in presentation_keywords)
        is_recommendation = any(keyword in question_lower for keyword in recommendation_keywords)
        
        # Strip YouTube URLs from the search query to prevent accidental recommendation triggers
        url_pattern = r'https?://(?:www\.)?youtube\.com/watch\?v=[0-9A-Za-z_-]{11}|https?://youtu\.be/[0-9A-Za-z_-]{11}'
        search_query = re.sub(url_pattern, '', question_lower)
        
        # Clean query for retrieval/search
        filler_phrases = presentation_keywords + recommendation_keywords + [
            "can you", "i want to learn", "give me", "article about", "about", 
            "some", "best", "please", "based on the course materials", "based on",
            "from the materials", "summarize", "explain", "تلخيص", "شرح", "وضوح",
            "رشح", "ممكن", "عايز", "اتعلم", "عن", "افضل", "أفضل", "لي", "اعطني"
        ]
        for phrase in filler_phrases:
            search_query = search_query.replace(phrase, "")
        
        # Clean redundant spacing and punctuation
        search_query = " ".join(search_query.split())
        search_query = search_query.strip(" ?!.،؟")
        
        # Fallback to full question (with URL stripped) if cleaning stripped everything
        if not search_query or len(search_query) < 2:
            search_query = re.sub(url_pattern, '', question).strip()
            if not search_query: search_query = "general"
            
        logger.info(f"Cleaned search query for retrieval: '{search_query}'")

        # 1. Fetch YouTube transcript if a URL is provided
        youtube_data = self.youtube_processor.process_url(question)
        youtube_transcript = youtube_data.get("transcript") if youtube_data else None
        video_meta = {
            "title": youtube_data.get("title"),
            "duration": youtube_data.get("duration"),
            "video_id": youtube_data.get("video_id")
        } if youtube_data else None
        
        # 2. Standard RAG process (Course Materials)
        try:
            # Use the cleaned search_query for much better retrieval
            raw_documents = self.retriever.retrieve(search_query)
            logger.info(f"Retrieved {len(raw_documents)} raw documents")
            
            # Filter out assessment/question-based documents
            documents = []
            assessment_keywords = [
                "which of the following", "question", "solve", 
                "a)", "b)", "c)", "d)", 
                "اختر", "أي مما يلي"
            ]
            
            for doc in raw_documents:
                content_lower = doc.page_content.lower()
                is_assessment = any(keyword in content_lower for keyword in assessment_keywords)
                
                if not is_assessment:
                    documents.append(doc)
                else:
                    logger.info(f"Filtered out assessment document: {doc.page_content[:50]}...")
            
            logger.info(f"Remaining documents after filtering: {len(documents)}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            documents = []

        # 3. Create structured hybrid context
        context_parts = []
        
        if youtube_data:
            meta_header = f"[VIDEO_TITLE: {video_meta['title']}]\n[VIDEO_DURATION: {video_meta['duration']}]\n"
            raw_transcript = youtube_data.get("transcript")
            
            if raw_transcript and "[ERROR:" not in str(raw_transcript):
                content = raw_transcript
            elif raw_transcript and "[ERROR:" in str(raw_transcript):
                content = f"[Transcription Blocked: {raw_transcript}]"
            else:
                content = "[No Transcript Available. Suggest installing ffmpeg for local transcription.]"
            
            context_parts.append("[SOURCE: YOUTUBE_VIDEO_TRANSCRIPT]\n" + meta_header + content)
        
        if documents:
            course_text = "\n\n".join([doc.page_content for doc in documents])
            context_parts.append("[SOURCE: OFFICIAL_COURSE_MATERIALS]\n" + course_text)
        
        full_context = "\n\n" + "\n\n---\n\n".join(context_parts) if context_parts else ""
        
        recommendation_data = None
        if is_recommendation:
            logger.info("Recommendation intent detected, fetching additional resources...")
            
            # Smarter recommendation query: If cleaning left us with nothing, use the video title
            rec_query = search_query
            if (not rec_query or rec_query == "general") and video_meta and video_meta['title'] != "Unknown Title":
                rec_query = video_meta['title']
                logger.info(f"Using video title for recommendations: '{rec_query}'")
            
            recommendation_data = self.recommender.get_all_recommendations(rec_query)
            
            yt_count = len(recommendation_data.get('youtube', []))
            logger.info(f"Found {yt_count} YouTube recommendations for '{rec_query}'")

        # 4.5 Handle Presentation intent
        if is_presentation:
            logger.info("Presentation intent detected...")
            
            # Combine available context to use as source for slides
            slide_source_content = full_context if full_context else question
            
            # Get structured slides from LLM
            logger.info("Structuring slides using LLM...")
            slides_data = self.generator.get_presentation_structure(slide_source_content)
            
            # Scan for local 'uploaded' images
            image_dir = os.path.join("data", "presentation_images")
            local_images = []
            if os.path.exists(image_dir):
                # Get common image extensions
                extensions = ('.png', '.jpg', '.jpeg', '.webp')
                local_images = [
                    os.path.join(image_dir, f) for f in os.listdir(image_dir)
                    if f.lower().endswith(extensions)
                ]
                local_images.sort() # Ensure consistent order
                logger.info(f"Found {len(local_images)} local images for the presentation")
            
            # Create the presentation with local images
            filename = "generated_presentation.pptx"
            pptx_path = self.presentation_maker.create_presentation(slides_data, local_images, filename)
            
            # Detect language for the message
            has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
            
            if pptx_path:
                if has_arabic:
                    msg = f"لقد قمت بإنشاء العرض التقديمي لك. يمكنك العثور عليه هنا: {pptx_path}"
                else:
                    msg = f"I have created a presentation for you. You can find it here: {pptx_path}"
                return {
                    "answer": msg,
                    "sources": [],
                    "presentation_path": pptx_path
                }
            else:
                if has_arabic:
                    msg = "حاولت إنشاء عرض تقديمي ولكن حدث خطأ ما أثناء إنشاء الملف."
                else:
                    msg = "I tried to create a presentation but something went wrong during the file generation."
                return {
                    "answer": msg,
                    "sources": []
                }

        # 5. Handle empty context return - only if NO recommendations either
        if not context_parts and not recommendation_data:
            return {
                "answer": "لم أتمكن من العثور على معلومات ذات صلة في المواد الدراسية ولم أجد ترشيحات خارجية." if has_arabic else "I couldn't find any relevant information in the video or course materials, and no recommendations were found.",
                "sources": []
            }
        
        # full_context already defined above
        
        # 6. Generate answer with history
        answer = self.generator.generate_answer(
            question, 
            full_context, 
            is_youtube=bool(youtube_data), 
            history=history,
            recommendations=recommendation_data
        )
        
        # Prepare sources for UI
        sources = []
        if youtube_data:
            sources.append({
                "content": f"Video: {video_meta['title']} ({video_meta['duration']})", 
                "metadata": {"source": "YouTube", "video_id": video_meta['video_id']}
            })
        
        for doc in documents:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context": full_context
        }
    
