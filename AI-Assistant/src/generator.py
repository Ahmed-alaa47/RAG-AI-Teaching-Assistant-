from typing import Dict, Optional
import logging
import re
import json
from config.settings import USE_GROQ, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Install with: pip install groq")


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
        "en": """Provide a well-structured, professional explanation using the following format:
- Start with a brief, clear definition or summary (1-2 sentences).
- Use markdown headers (###, ####) to organize sections logically.
- Use bullet points with **bold** labels for key characteristics or points.
- Include code blocks only if code exists in the provided materials.
- End with a concise one-sentence summary.
- Do NOT use "Step 1, Step 2" format. Do NOT number every paragraph.""",
        "ar": """قدم شرحاً منظماً واحترافياً بالتنسيق التالي:
- ابدأ بتعريف موجز وواضح (جملة أو جملتان).
- استخدم عناوين markdown لتنظيم الأقسام بشكل منطقي.
- استخدم النقاط مع تسميات굵게 للخصائص والنقاط الرئيسية.
- أضف كتل الكود فقط إذا كان الكود موجوداً في المواد المقدمة.
- اختم بجملة ملخصة موجزة.
- لا تستخدم تنسيق الخطوات المرقمة."""
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
    Handles the generation of answers using Groq API (cloud LLM) or fallback methods.
    """

    def __init__(self):
        self.model = GROQ_MODEL
        self.api_key = GROQ_API_KEY

        print(f"DEBUG: Generator initialized with model='{self.model}' via Groq API")

        if GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)
            print("DEBUG: Groq client initialized successfully.")
        else:
            self.client = None
            if not GROQ_AVAILABLE:
                print("DEBUG: Groq library NOT available. Run: pip install groq")
            if not self.api_key:
                print("DEBUG: GROQ_API_KEY is missing. Check your .env file.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        context: str,
        question_type: str = None,
        is_youtube: bool = False,
        history: list = None,
        recommendations: dict = None,
    ) -> str:
        """Generate an answer from context using Groq or a simple fallback."""

        if question_type is None:
            question_type = self._detect_question_type(question)

        if USE_GROQ and GROQ_AVAILABLE and self.client:
            try:
                return self._generate_with_groq(
                    question, context, question_type, is_youtube, history, recommendations
                )
            except Exception as e:
                logger.error(f"Groq generation failed: {e}")
                logger.info("Falling back to simple context display")

        # Fallback
        return f"Based on the course materials:\n\n{context[:800]}..."

    def get_presentation_structure(self, content: str, title: str = "Presentation") -> list:
        """
        Uses the LLM to structure content into a list of slides for a presentation.
        Returns: [{"title": "...", "content": ["...", "..."]}, ...]
        """
        if not USE_GROQ or not GROQ_AVAILABLE or not self.client:
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
            )
            text_response = response.choices[0].message.content.strip()
            logger.info(f"LLM Structure Response: {text_response[:500]}...")

            json_match = re.search(r'\[\s*{.*}\s*\]', text_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(text_response)

        except Exception as e:
            logger.error(f"Failed to structure presentation: {e}")
            return [{"title": title, "content": ["Error structuring content into slides."]}]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_question_type(self, question: str) -> str:
        """Automatically detect question type based on heuristics."""
        question_lower = question.lower().strip()

        if re.search(r"\b[A-D]\)", question) or any(
            kw in question_lower for kw in ["choose", "which of the following", "اختر", "أي مما يلي"]
        ):
            return "mcq"

        if any(kw in question_lower for kw in ["true or false", "صح أم خطأ", "صح أو خطأ"]):
            return "true_false"

        if "____" in question or "..." in question or any(
            kw in question_lower for kw in ["complete", "fill in", "اكمل", "أكمل"]
        ):
            return "fill_blank"

        code_keywords = [
            "code", "function", "variable", "class", "loop", "algorithm",
            "complexity", "syntax", "debug", "refactor",
            "كود", "دالة", "متغير", "خوارزمية",
        ]
        if "```" in question or any(kw in question_lower for kw in code_keywords):
            return "code"

        return "explain"

    def _build_messages(
        self,
        question: str,
        context: str,
        question_type: str,
        history: list,
        rec_text: str,
        has_arabic: bool,
    ) -> list:
        """Build the messages list for the Groq chat completion call."""

        q_type = question_type if question_type in QUESTION_TYPE_INSTRUCTIONS else "explain"
        instruction = QUESTION_TYPE_INSTRUCTIONS[q_type]["ar" if has_arabic else "en"]

        # ── System prompt ──────────────────────────────────────────────
        if has_arabic:
            system_prompt = f"""أنت مساعد تعليمي. معرفتك مقيدة تماماً بالمحتوى الموجود في [SOURCE: OFFICIAL_COURSE_MATERIALS] و [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT] فقط.

قواعد صارمة يجب اتباعها دون استثناء:
1. ⚠️ ممنوع الإجابة من معرفتك العامة الخاصة. لا يُسمح لك باستخدام أي معلومة خارج المواد الدراسية المقدمة.
2. إذا لم يُذكر الموضوع أو الإجابة في المحتوى المقدم، يجب أن تجيب فقط بـ:
   "⚠️ هذا الموضوع غير مذكور في المواد الدراسية. لا أستطيع الإجابة إلا بناءً على المستندات المقدمة."
   لا تحاول الإجابة أو التخمين أو تقديم معلومات جزئية من خارج المواد.
3. الاستثناء الوحيد للقاعدة الأولى هو إذا كان السؤال عن بناء الجملة البرمجية (Syntax) أو تصحيح أخطاء كود موجود فعلاً في المواد الدراسية.
4. إذا سأل الطالب عن معلومة وقدم رابط فيديو، ابحث أولاً في [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT] واذكر الوقت الدقيق (مثال: "ذكر المحاضر عند الدقيقة [00:12:30] أن...").
5. في حالة ملاحظة "[Transcription Blocked]" أو "[No Transcript Available]"، أخبر الطالب أنك لم تتمكن من قراءة التفاصيل واعرض المساعدة من مواد الكورس.
6. في حالة الأسئلة التي تطلب "ترشيحات"، استخدم بيانات [RECOMMENDED_RESOURCES] فقط.
7. لا تعرض الترشيحات إذا كان الطالب يطلب تلخيص الفيديو المقدم.
8. ممنوع تماماً تقديم أي روابط بحث عامة أو روابط لمواقع أخرى.
9. استخدم سجل المحادثة لفهم الأسئلة المتابعة.
10. الإجابة يجب أن تكون منسقة ومنظمة (استخدم النقاط والعناوين الفرعية والكود عند الحاجة).
11. إذا كانت قائمة الترشيحات فارغة، قل فقط: "عذراً، لم أجد روابط يوتيوب مناسبة حالياً."

تعليمات خاصة بنوع السؤال:
{instruction}"""
        else:
            system_prompt = f"""You are a helpful teaching assistant. Your knowledge is STRICTLY LIMITED to the content provided in [SOURCE: OFFICIAL_COURSE_MATERIALS] and [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT] below.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. ⚠️ NEVER answer from your own general knowledge. You are NOT allowed to use any information outside the provided course materials.
2. If the topic or answer is NOT found in the provided content, you MUST respond ONLY with:
   "⚠️ This topic is not covered in the course materials. I can only answer questions based on the provided documents."
   Do NOT attempt to answer, guess, or provide partial information from outside the materials.
3. The ONLY exception to rule 1 is if the question is about CODE SYNTAX or DEBUGGING of code that already appears in the course materials — in that case you may assist technically.
4. Answer from [SOURCE: YOUTUBE_VIDEO_TRANSCRIPT] first if a video link was provided. Include exact timestamps (e.g., "The speaker mentions at [00:05:20] that...").
5. If you see "[Transcription Blocked]" or "[No Transcript Available]", inform the user and offer to help using course materials instead.
6. If the student asks for "recommendations" or "resources", use ONLY the [RECOMMENDED_RESOURCES] section.
7. NEVER show recommendations when the user asks to summarize the provided video.
8. DO NOT provide general search links or links to other platforms.
9. Use conversation history to understand follow-up questions.
10. Format all responses with markdown (bullet points, subheadings, code blocks where needed).
11. If the recommendations list is empty, say: "I'm sorry, I couldn't find any specific YouTube recommendations for this topic at the moment."

Special instructions for this question type:
{instruction}"""

        # ── User content ───────────────────────────────────────────────
        if has_arabic:
            user_content = f"""المحتوى المقدم:
{context if context.strip() else "لا يوجد محتوى إضافي من المصادر."}

[RECOMMENDED_RESOURCES]
{rec_text if rec_text else "لا توجد ترشيحات متوفرة حالياً."}

سؤال الطالب: {question}"""
        else:
            user_content = f"""Provided Content:
{context if context.strip() else "No additional context from sources."}

[RECOMMENDED_RESOURCES]
{rec_text if rec_text else "No specific recommendations found currently."}

Student Question: {question}"""

        # ── Assemble messages (system → history → current user turn) ───
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for turn in history[-5:]:   # last 5 turns for context window efficiency
                role = "user" if turn["role"] == "user" else "assistant"
                messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate_with_groq(
        self,
        question: str,
        context: str,
        question_type: str,
        is_youtube: bool,
        history: list = None,
        recommendations: dict = None,
    ) -> str:
        """Generate an answer using the Groq cloud API."""

        has_arabic = any("\u0600" <= ch <= "\u06FF" for ch in question)

        # Format recommendations
        rec_text = ""
        if recommendations and recommendations.get("youtube"):
            rec_text = "YouTube Courses & Videos:\n"
            for rec in recommendations["youtube"]:
                rec_text += f"- {rec['title']} ({rec['duration']}): {rec['link']}\n"

        messages = self._build_messages(
            question, context, question_type, history or [], rec_text, has_arabic
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.5,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq generation failed in _generate_with_groq: {e}")
            return "Error: Failed to generate a response using the Groq API."