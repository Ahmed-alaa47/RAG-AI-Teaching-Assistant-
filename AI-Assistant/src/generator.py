from typing import Dict, Optional
import logging
from config.settings import USE_OLLAMA, OLLAMA_MODEL, OLLAMA_BASE_URL

# Basic logging setup for this module (if needed, or rely on root logger)
logger = logging.getLogger(__name__)

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not installed. Install with: pip install ollama")

QUESTION_TYPE_INSTRUCTIONS = {
    "mcq": {
        "en": "Select the correct option from the provided choices.",
        "ar": "اختر الإجابة الصحيحة من الخيارات المقدمة."
    },
    "fill_blank": {
        "en": "Provide only the missing word or phrase to complete the statement.",
        "ar": "أعط الكلمة أو العبارة المفقودة فقط لإكمال الجملة."
    },
    "explain": {
        "en": "Explain the answer clearly and in detail.",
        "ar": "اشرح الإجابة بوضوح وبالتفصيل."
    },
    "true_false": {
        "en": "State whether the statement is True or False, followed by a brief explanation.",
        "ar": "اذكر ما إذا كانت العبارة صحيحة أم خاطئة، متبوعاً بشرح موجز."
    }
}

class Generator:
    """
    Handles the generation of answers using an LLM (Ollama) or fallback methods.
    """
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
        print(f"DEBUG: Generator initialized with model='{self.model}' and base_url='{self.base_url}'")
        # Initialize client with host
        if OLLAMA_AVAILABLE:
            self.client = ollama.Client(host=self.base_url)
        else:
            print("DEBUG: Ollama library NOT available")
            self.client = None

    def generate_answer(self, question: str, context: str, question_type: str = None, is_youtube: bool = False, history: list = None) -> str:
        """Generate answer from context using Ollama or simple extraction."""
        
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = self._detect_question_type(question)
            
        # Check if Ollama should be used and is available
        if USE_OLLAMA and OLLAMA_AVAILABLE:
            try:
                return self._generate_with_ollama(question, context, question_type, is_youtube, history)
            except Exception as e:
                logger.error(f"Ollama generation failed: {e}")
                logger.info("Falling back to simple context display")
        
        # Fallback: Simple context display
        return f"Based on the course materials:\n\n{context[:800]}..."

    def _detect_question_type(self, question: str) -> str:
        """
        Automatically interprets the question type based on heuristics.
        """
        question_lower = question.lower().strip()
        
        # 1. Check for MCQ
        # Look for "A)", "B)", etc. or specific keywords
        import re
        if re.search(r"\b[A-D]\)", question) or \
           any(keyword in question_lower for keyword in ["choose", "which of the following", "اختر", "أي مما يلي"]):
            return "mcq"
            
        # 2. Check for True/False
        if any(keyword in question_lower for keyword in ["true or false", "صح أم خطأ", "صح أو خطأ"]):
            return "true_false"
            
        # 3. Check for Fill in the blank
        if "____" in question or "..." in question or \
           any(keyword in question_lower for keyword in ["complete", "fill in", "اكمل", "أكمل"]):
            return "fill_blank"
            
        # 4. Default to Explain
        return "explain"

    def _generate_with_ollama(self, question: str, context: str, question_type: str, is_youtube: bool, history: list = None) -> str:
        """Generate answer using Ollama local LLM with language detection and history."""
        
        # Detect if question contains Arabic characters
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
        
        # Get specific instruction or fallback to explain
        q_type = question_type if question_type in QUESTION_TYPE_INSTRUCTIONS else "explain"
        instruction = QUESTION_TYPE_INSTRUCTIONS[q_type]["ar" if has_arabic else "en"]

        # Format history string
        history_text = ""
        if history:
            history_text = "\n\n[CONVERSATION_HISTORY]\n"
            for turn in history[-5:]:  # Last 5 turns
                role = "User" if turn['role'] == 'user' else "Assistant"
                history_text += f"{role}: {turn['content']}\n"
            history_text += "---\n"

        if has_arabic:
            prompt = f"""أنت مساعد تعليمي ذكي. سأزودك بمحتوى مقسم حسب المصدر أدناه. {history_text}
تعليمات هامة جداً:
1. إذا سأل الطالب عن معلومة وقدم رابط فيديو، ابحث أولاً في [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT].
2. إذا وجدت الإجابة في الفيديو، اذكرها واستخدم الطوابع الزمنية الموجودة (مثلاً [HH:MM:SS]).
3. إذا لم تجد الإجابة في الفيديو ولكنها موجودة في [SOURCE: OFFICIAL_COURSE_MATERIALS]، قدم الإجابة وابدأ جملتك بـ: "هذه المعلومة غير مذكورة في الفيديو، ولكن بناءً على المواد الدراسية..."
4. إذا لم تجد الإجابة في أي من المصادر، قل فقط: "الإجابة غير موجودة في المحتوى المقدم." ولا تحاول الإجابة من معلوماتك الخارجية.
5. لا تعتمد على الأسئلة أو التمارين كمصدر للتعريفات.
6. استخدم تاريخ المحادثة (History) أعلاه إذا كان الطالب يطرح أسئلة متابعة (مثل "ماذا أيضاً؟" أو "اشرح ذلك").

تعليمات نوع السؤال: {instruction}

المحتوى المقدم:
{context}

سؤال الطالب: {question}

الإجابة:"""
        else:
            prompt = f"""You are a helpful teaching assistant. I will provide you with content separated by source below. {history_text}
IMPORTANT INSTRUCTIONS:
1. If the student provides a link/video, attempt to answer from [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT] first.
2. If the answer is found in the video, include it and use the provided timestamps (e.g., [HH:MM:SS]) for citation.
3. If the answer is NOT in the video transcript but IS present in [SOURCE: OFFICIAL_COURSE_MATERIALS], provide the answer but explicitly start with: "This information was not mentioned in the video, but according to the course materials..."
4. If the answer is missing from BOTH sources, respond strictly with: "The answer is not available in the provided material."
5. Do NOT use outside knowledge. Do NOT use assessment questions as sources for theory.
6. Use the [CONVERSATION_HISTORY] provided above to understand follow-up questions or references (like "What about...?" or "Tell me more").

Question Type Instructions: {instruction}

Provided Content:
{context}

Student Question: {question}

Answer:"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama error: {e}. Make sure Ollama is installed and '{self.model}' model is downloaded.")
