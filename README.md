# 🎓 Course Material AI Assistant (مساعد المواد الدراسية الذكي)

A powerful **RAG-based** (Retrieval-Augmented Generation) AI teaching assistant that helps students and educators interact with multi-format course materials through a unified chat interface. The system supports **Arabic and English**, processes documents, images, and YouTube videos, and exposes all features as **REST APIs** for front-end integration.

---

## 🚀 Features

### Core AI Capabilities
- **Smart Q&A** — Ask questions about your course materials and get AI-generated answers with source references.
- **YouTube Video Analysis** — Paste a YouTube URL and ask questions. The system extracts the transcript and answers from the video content.
- **Recommendations** — Ask for learning recommendations and get curated YouTube video suggestions.
- **Presentation Generation** — Ask the AI to create a PowerPoint presentation on any topic from your course materials.
- **Document & Image Understanding** — Upload PDFs, DOCX, PPTX, images, and more. The system extracts text (including OCR for scanned documents) and answers questions.
- **Conversation History** — The system remembers previous messages within a session for follow-up questions.

### Technical Highlights
- **Multilingual Support** — Fully optimized for Arabic and English using `paraphrase-multilingual-MiniLM-L12-v2` embeddings.
- **Local AI** — Runs 100% locally using [Ollama](https://ollama.com/) (LLaMA 3.1 8B), ensuring data privacy and zero API costs.
- **OCR** — Tesseract-based text extraction from scanned PDFs and standalone images (supports Arabic + English).
- **Smart Filtering** — Automatically filters out assessment-style content (MCQs) to provide cleaner context.
- **REST API** — Full Django Rest Framework (DRF) server for seamless front-end integration.

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or 3.11 |
| **Ollama** | Download from [ollama.com](https://ollama.com/). Then pull the model: `ollama pull llama3.1:8b` |
| **Tesseract OCR** | **Windows**: Install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) (include Arabic language data). **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-ara` |
| **Poppler** *(optional, for PDF images)* | **Linux**: `sudo apt install poppler-utils`. **Windows**: Download poppler and add `bin/` to PATH. |
| **ffmpeg** *(optional, for YouTube fallback)* | Required only if YouTube videos don't have transcripts and need Whisper transcription. |

---

## 🛠️ Installation (Step-by-Step)

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmed-alaa47/RAG-AI-Teaching-Assistant-.git
cd RAG-AI-Teaching-Assistant-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
# Windows (PowerShell):
venv\Scripts\activate

# Windows (CMD):
venv\Scripts\activate.bat

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This will install PyTorch (CPU version), sentence-transformers, Django, Django REST Framework, and all other dependencies. The first run will also download the embedding model (~130MB).

### 4. Install & Start Ollama

1. Download Ollama from [ollama.com](https://ollama.com/) and install it.
2. Pull the LLaMA 3.1 model:

```bash
ollama pull llama3.1:8b
```

3. Make sure Ollama is running (it starts automatically on install, or run `ollama serve`).

### 5. Add Your Course Materials

Place your course files in the `AI-Assistant/data/raw/` directory:

```
AI-Assistant/
  data/
    raw/          ← Put your files here
      lecture1.pdf
      notes.docx
      slides.pptx
      diagram.png
```

**Supported formats:** PDF, DOCX, PPTX, TXT, PNG, JPG, JPEG

### 6. (Optional) Add Presentation Images

If you want images in your generated presentations, place them in:

```
AI-Assistant/
  data/
    presentation_images/    ← Put images here
      slide1.png
      slide2.jpg
```

---

## 🚀 Running the Project

### Option A: Run the API Server (Recommended for front-end integration)

```bash
cd AI-Assistant
python manage.py runserver 0.0.0.0:8000
```

The server will:
1. Automatically load the existing vector store on startup (if available).
2. Serve the API at **http://localhost:8000**
3. Provide a Browsable API at the endpoint URLs (if accessed via browser)

> **First time?** After starting the server, call `POST /initialize` to process your course materials and build the vector store. This only needs to be done once (or when you add new files).

### Option B: Run the CLI (Terminal-based chat)

```bash
cd AI-Assistant
python main.py
```

---

## 🌐 API Reference

The API server provides the following endpoints. The primary endpoint is `/chat` — it handles **all features automatically** through intelligent intent detection. The other endpoints are available for direct access if the front-end needs them.

### Primary Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | **Unified chat** — send any question and the AI automatically detects intent (Q&A, YouTube, recommendations, presentation) |

The `/chat` endpoint accepts:
```json
{
  "question": "Your question here (can include a YouTube URL)",
  "history": [
    {"role": "user", "content": "previous question"},
    {"role": "assistant", "content": "previous answer"}
  ]
}
```

**How intent detection works in `/chat`:**

| What the user types | Auto-detected intent | What happens |
|---|---|---|
| `"What is a stack?"` | Q&A | Searches course materials and answers |
| `"https://youtube.com/watch?v=xxx What is this about?"` | YouTube Q&A | Extracts transcript, answers from video |
| `"Recommend me courses about Python"` | Recommendation | Fetches YouTube video recommendations |
| `"Create a presentation about linked lists"` | Presentation | Generates a .pptx file |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns API status and document count |
| `POST` | `/initialize` | Process course materials and build the vector store |

### Utility Endpoints (Direct access)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/youtube/process` | Process a YouTube URL directly. Accepts optional `question` field |
| `POST` | `/recommendations` | Get YouTube recommendations for a topic |
| `POST` | `/presentation/create` | Generate a PowerPoint presentation for a topic |
| `GET` | `/presentation/download/{filename}` | Download a generated .pptx file |
| `POST` | `/documents/upload` | Upload a course material file (indexes it into the vector store) |
| `POST` | `/documents/ask` | Upload a file AND ask a question about it in one call |
| `POST` | `/images/upload` | Upload an image for use in generated presentations |

---

## 🐳 Docker Support

### Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build the container with all dependencies (including Tesseract OCR).
- Start the API server on **port 8000**.
- Connect to your local Ollama instance via `host.docker.internal`.

> **Important:** Ollama must be running on your host machine. Docker connects to it automatically.

### Run with Docker only

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 rag-assistant
```

---

## 📁 Project Structure

```
RAG-AI-Teaching-Assistant-/
├── AI-Assistant/
│   ├── manage.py                 # Django management script
│   ├── config_proj/              # Django site configuration
│   ├── api_app/                  # Django REST API application
│   ├── models.py                 # Pydantic request/response schemas
│   ├── main.py                   # CLI entry point
│   ├── config/
│   │   └── settings.py           # All configuration (paths, models, thresholds)
│   ├── data/
│   │   ├── raw/                  # ← Place course materials here
│   │   ├── processed/            # Vector store & cache (auto-generated)
│   │   └── presentation_images/  # ← Place images for presentations here
│   ├── src/
│   │   ├── rag_pipeline.py       # Main orchestrator (intent detection + query routing)
│   │   ├── document_processor.py # Loads & chunks documents (PDF, DOCX, PPTX, images)
│   │   ├── youtube_processor.py  # YouTube transcript extraction (API + Whisper fallback)
│   │   ├── generator.py          # LLM answer generation via Ollama
│   │   ├── recommender.py        # YouTube video recommendations
│   │   ├── presentation_maker.py # PowerPoint (.pptx) generation
│   │   ├── vector_store.py       # ChromaDB vector store management
│   │   ├── retriever.py          # Similarity-based document retrieval
│   │   └── embeddings.py         # Sentence-transformer embedding manager
│   ├── utils/
│   │   └── helpers.py            # Arabic text display + formatting utilities
│   └── presentations/            # Generated presentations output
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Configuration

All settings are in `AI-Assistant/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1:8b` | The LLM model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model |
| `CHUNK_SIZE` | `800` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Number of documents to retrieve |
| `ENABLE_OCR` | `True` | Enable/disable image text extraction |

---

## 🧪 Quick Test

After starting the API server, you can test it immediately:

```bash
# Health check
curl http://localhost:8000/health

# Initialize (first time only)
curl -X POST http://localhost:8000/initialize

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a linked list?"}'

# Ask about a YouTube video
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "https://youtube.com/watch?v=VIDEO_ID Explain what this video covers"}'

# Get recommendations
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Recommend me courses about data structures"}'
```

Or open the endpoints in your browser to view the DRF browsable API.
