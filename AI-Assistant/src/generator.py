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
    },
    "code": {
        "en": """You are a technical mentor and programming expert. 
If the student asks to write, implement, or provide code:
- Provide the complete, functional, and well-commented code implementation first.
- Then, provide a structured analysis including the sections below.

If the student provides code to be analyzed:
- Provide a structured response including:
1. High-level Overview: What does this code do in simple terms?
2. Step-by-Step Walkthrough: How does it execute?
3. Variables & Functions: Explain the key components.
4. Control Flow: Identify loops, conditionals, or recursion.
5. Complexity Analysis: Estimate Time and Space complexity.
6. Edge Cases & Potential Issues: What could go wrong?
7. Improvements: Suggest optimizations or better practices.
If the code has errors, point them out clearly.""",
        "ar": """أنت معلم تقني وخبير برمجة.
إذا طلب الطالب كتابة أو تنفيذ أو تقديم كود:
- قدم الكود البرمجي الكامل والوظيفي والمشروح جيداً أولاً.
- ثم قدم تحليلاً منظماً يتضمن الأقسام المذكورة أدناه.

إذا قدم الطالب كوداً للتحليل:
- قدم استجابة منظمة تشمل:
1. نظرة عامة: ماذا يفعل هذا الكود بلمحة سريعة؟
2. شرح خطوة بخطوة: كيف يتم التنفيذ؟
3. المتغيرات والدوال: شرح المكونات الرئيسية.
4. تدفق التحكم: تحديد الحلقات (loops)، الشروط، أو التكرار (recursion).
5. تحليل التعقيد: تقدير التعقيد الزمني والمكاني (Complexity analysis).
6. الحالات الحادة والمشاكل المحتملة: ما الذي قد يفشل؟
7. التحسينات: اقتراح تحسينات أو أفضل الممارسات.
إذا كان الكود يحتوي على أخطاء، وضحها بوضوح."""
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

    def generate_answer(self, question: str, context: str, question_type: str = None, is_youtube: bool = False, history: list = None, recommendations: dict = None) -> str:
        """Generate answer from context using Ollama or simple extraction."""
        
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = self._detect_question_type(question)
            
        # Check if Ollama should be used and is available
        if USE_OLLAMA and OLLAMA_AVAILABLE:
            try:
                return self._generate_with_ollama(question, context, question_type, is_youtube, history, recommendations)
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
            
        # 4. Check for Code
        code_keywords = [
            "code", "function", "variable", "class", "loop", "algorithm", "complexity", 
            "syntax", "debug", "refactor", "كود", "دالة", "متغير", "خوارزمية"
        ]
        if "```" in question or any(keyword in question_lower for keyword in code_keywords):
            return "code"
            
        # 5. Default to Explain
        return "explain"

    def _generate_with_ollama(self, question: str, context: str, question_type: str, is_youtube: bool, history: list = None, recommendations: dict = None) -> str:
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
        
        # Format recommendations if available
        rec_text = ""
        if recommendations and recommendations.get('youtube'):
            rec_text = "\n\n[RECOMMENDED_RESOURCES]\n"
            rec_text += "YouTube Courses & Videos:\n"
            for rec in recommendations['youtube']:
                rec_text += f"- {rec['title']} ({rec['duration']}): {rec['link']}\n"
            rec_text += "---\n"

        if has_arabic:
            prompt = f"""أنت مساعد تعليمي ذكي وخبير في البرمجة. سأزودك بمحتوى مقسم حسب المصدر أدناه (إذا وجد). {history_text} {rec_text}
تعليمات هامة جداً:
1. إذا كان السؤال عن الكود، يجب عليك التصرف كمعلم تقني واستخدام معرفتك العميقة بالبرمجة لشرح الكود خطوة بخطوة.
2. إذا سأل الطالب عن معلومة وقدم رابط فيديو، ابحث أولاً في [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT]. إذا وجدته، يجب أن يكون هو المصدر الأساسي والوحيد للإجابة ما لم يُطلب غير ذلك.
3. إذا وجدت الإجابة في محتوى الفيديو، يجب عليك وبدون استثناء ذكر الوقت الدقيق الذي وردت فيه المعلومة باستخدام الطابع الزمني المزوّد (مثال: "ذكر المحاضر في فيديو '[العنوان]' عند الدقيقة [00:12:30] أن...").
4. في حالة ملاحظة "[Transcription Blocked]" أو "[No Transcript Available]" مع وجود عنوان للفيديو، أخبر الطالب أنك حصلت على عنوان الفيديو ولكن لم تتمكن من قراءة التفاصيل، واعرض المساعدة باستخدام مواد الكورس الأخرى.
5. في حالة الأسئلة التي تطلب "ترشيحات" أو "كورسات" (Recommendations)، يجب عليك فوراً وبشكل أساسي استخدام البيانات الموجودة في [RECOMMENDED_RESOURCES] وعرضها كروابط يوتيوب مباشرة.
6. لا تعرض الترشيحات (Recommendations) أبداً إذا كان الطالب يطلب تلخيص الفيديو المزوّد (Summarize)؛ الترشيحات تكون فقط عند طلب "مصادر إضافية" أو "مزيد من المعلومات".
7. ممنوع تماماً تقديم أي روابط بحث عامة أو روابط لمواقع أخرى غير اليوتيوب المزوّدة في [RECOMMENDED_RESOURCES].
6. لا تبحث في محتوى الدروس ([SOURCE: OFFICIAL_COURSE_MATERIALS]) عن ترشيحات عامة إذا كان الطالب يطلب مصادر خارجية؛ استخدم فقط [RECOMMENDED_RESOURCES].
7. ممنوع تماماً تقديم أي روابط بحث عامة أو روابط لمواقع أخرى غير اليوتيوب المزوّدة في [RECOMMENDED_RESOURCES].
8. الإجابة يجب أن تبدأ بترشيحات اليوتيوب بشكل واضح جداً (العنوان، المدة، والرابط).
9. لا تتجاهل روابط اليوتيوب أبداً إذا كانت متوفرة.
10. الإجابة يجب أن تكون منسقة ومنظمة بشكل جيد (استخدم النقاط، العناوين الفرعية، والجداول إذا لزم الأمر).
11. التحذير النهائي: إذا وجدت "لا توجد ترشيحات متوفرة حالياً" أو كانت القائمة فارغة، **ممنوع نهائياً** اقتراح أي كورسات أو قنوات من معرفتك الخاصة. في هذه الحالة قل فقط: "عذراً، لم أجد روابط يوتيوب مناسبة حالياً."

تعليمات خاصة:
{instruction}

المحتوى المقدم:
{context if context.strip() else "لا يوجد محتوى إضافي من المصادر."}

[RECOMMENDED_RESOURCES]
{rec_text if rec_text else "لا توجد ترشيحات متوفرة حالياً."}

سؤال الطالب: {question}

الإجابة:"""
        else:
            prompt = f"""You are a helpful teaching assistant and an experienced programming mentor. I will provide you with content separated by source below (if available). {history_text}
IMPORTANT INSTRUCTIONS:
1. If the question is code-related, act as a technical mentor and use your deep programming knowledge to explain the code thoroughly.
2. If the student provides a link/video, attempt to answer from [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT] first. This is your PRIMARY source.
3. If the answer is found in the video transcript, you MUST include the exact timestamp for each point discussed (e.g., "The speaker mentions in the video '[Title]' at [00:05:20] that...").
4. If you see "[Transcription Blocked]" or "[No Transcript Available]" but have a Video Title, inform the user that you identified the video but couldn't access its internal content, then offer to help using course materials.
5. If the student asks for "recommendations", "courses", or "resources", prioritize the data in [RECOMMENDED_RESOURCES] and present them as direct links.
6. NEVER show recommendations if the user is asking to summarize the provided video link, unless they specifically ask for "more" or "alternative" resources.
6. DO NOT search through [SOURCE: OFFICIAL_COURSE_MATERIALS] for general recommendations if the student is asking for learning resources; use only the provided [RECOMMENDED_RESOURCES] section.
7. Show the YouTube videos with their Title, Duration, and clickable direct Links.
8. DO NOT provide general search links or mention other platforms. Only show the YouTube videos listed in [RECOMMENDED_RESOURCES].
9. NEVER ignore the provided YouTube links if they are available.
10. For code-related queries, you ARE allowed to use your general programming knowledge to explain syntax, complexity, and best practices. For general theory questions, stick to the provided materials.
11. If the answer is missing from BOTH sources and the question is NOT code-related and NOT a request for recommendations, respond strictly with: "The answer is not available in the provided material."
12. Use the [CONVERSATION_HISTORY] provided above to understand follow-up questions or references.
13. Ensure the response is well-formatted using markdown (bullet points, subheadings, etc.).

Special Instructions:
{instruction}

Provided Content:
{context if context.strip() else "No additional context from sources."}

[RECOMMENDED_RESOURCES]
{rec_text if rec_text else "No specific recommendations found currently."}

FINAL WARNING: If you see "No specific recommendations found currently." or if the list is empty, DO NOT suggest any channels, courses, OR links from your own knowledge. Say: "I'm sorry, I couldn't find any specific YouTube recommendations for this topic at the moment."

Student Question: {question}

Answer:"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            logger.error(f"Ollama generation failed in _generate_with_ollama: {e}")
            return "Error: Failed to generate a response using the local model."

    def get_presentation_structure(self, content: str, title: str = "Presentation") -> list:
        """
        Uses the LLM to structure content into a list of slides for a presentation.
        Returns a list of dictionaries: [{"title": "...", "content": ["...", "..."]}, ...]
        """
        if not USE_OLLAMA or not OLLAMA_AVAILABLE:
            # Fallback: Just create a single slide with the content
            return [{"title": title, "content": [content[:500] + "..."]}]

        prompt = f"""You are a presentation expert. Structure the following content into a sequence of professional PowerPoint slides.
Each slide must have:
1. 'title': A short, engaging title.
2. 'content': A list of 3-5 concise bullet points.

Return ONLY a valid JSON array of objects. Do not include any other text or explanation.

[FORMAT EXAMPLE]
[
  {{
    "title": "Introduction to AI", 
    "content": ["Definition of AI", "Brief history", "Core components"]
  }},
  {{
    "title": "Machine Learning", 
    "content": ["Types of ML", "Supervised learning", "Unsupervised learning"]
  }}
]

[CONTENT TO STRUCTURE]
{content}

JSON Response:"""

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt
            )
            text_response = response['response'].strip()
            
            # Debug tracking
            logger.info(f"LLM Structure Response: {text_response[:500]}...")
            
            # Extract JSON if LLM included extra text
            import json
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', text_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return json.loads(text_response)
        except Exception as e:
            logger.error(f"Failed to structure presentation: {e}")
            return [{"title": title, "content": ["Error structuring content into slides."]}]
