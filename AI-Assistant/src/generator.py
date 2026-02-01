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

    def generate_answer(self, question: str, context: str, question_type: str = None) -> str:
        """Generate answer from context using Ollama or simple extraction."""
        
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = self._detect_question_type(question)
            
        # Check if Ollama should be used and is available
        if USE_OLLAMA and OLLAMA_AVAILABLE:
            try:
                return self._generate_with_ollama(question, context, question_type)
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

    def _generate_with_ollama(self, question: str, context: str, question_type: str) -> str:
        """Generate answer using Ollama local LLM with language detection."""
        
        # Detect if question contains Arabic characters
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
        
        # Get specific instruction or fallback to explain
        q_type = question_type if question_type in QUESTION_TYPE_INSTRUCTIONS else "explain"
        
        if has_arabic:
            instruction = QUESTION_TYPE_INSTRUCTIONS[q_type]["ar"]
            # Arabic prompt
            prompt = f"""بناءً على المحتوى التعليمي التالي فقط، أجب على سؤال الطالب.
تعليمات الإجابة: {instruction}
لا تعتمد على الأسئلة أو التمارين كمصدر للتعريفات، واعتمد فقط على الشرح النظري.
إذا كانت الإجابة غير موجودة في المحتوى المقدم، قل فقط: "الإجابة غير موجودة في المحتوى المقدم." ولا تحاول التأليف.

المحتوى التعليمي:
{context}

سؤال الطالب: {question}

الإجابة:"""
        else:
            instruction = QUESTION_TYPE_INSTRUCTIONS[q_type]["en"]
            # English prompt
            prompt = f"""Based on the following course material ONLY, answer the student's question.
Answer Instructions: {instruction}
Do NOT use example questions, exercises, or assessments as a source of definitions. Prefer explanatory and theoretical sections only.
If the answer is not found in the context, respond strictly with: "The answer is not available in the provided material."

Course Material:
{context}

Student Question: {question}

Answer:"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama error: {e}. Make sure Ollama is installed and '{self.model}' model is downloaded.")
