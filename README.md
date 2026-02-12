# Course Material AI Assistant (مساعد المواد الدراسية الذكي)

A powerful RAG-based (Retrieval-Augmented Generation) AI assistant designed to help students and educators interact with multi-format course materials. The system supports both English and Arabic content and can process documents, images, and YouTube videos.

## 🚀 Features

- **Multilingual Support**: Fully optimized for Arabic and English processing using `paraphrase-multilingual-MiniLM-L12-v2`.
- **Hybrid Data Sources**:
  - **Documents**: Supports PDF, DOCX, PPTX, and TXT.
  - **Images**: Extracts text from standalone images (PNG, JPG, JPEG) and embedded images in PDFs.
  - **YouTube**: Fetches and indexes transcripts from YouTube URLs for video-based learning.
- **Advanced OCR**: Uses Tesseract OCR for high-quality text extraction from scanned documents and images.
- **Local AI**: Runs locally using Ollama, ensuring data privacy and no API costs.
- **Smart Filtering**: Automatically filters out assessment-style content (multiple choice questions) to provide cleaner context.
- **Conversation History**: Remembers context within a session for follow-up questions.

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

1.  **Python 3.10 or 3.11**
2.  **Ollama**: Download from [ollama.com](https://ollama.com/).
    - Pull the required model: `ollama pull llama3.1:8b`
3.  **Tesseract OCR**:
    - **Windows**: Install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Ensure you include **Arabic** language data.
    - **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-ara`
4.  **Poppler** (for PDF image processing):
    - **Linux**: `sudo apt install poppler-utils`
    - **Windows**: Download poppler for windows and add the `bin` folder to your PATH.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd RAG-AI-Teaching-Assistant-
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Project

1.  **Place your course materials**:
    Put your documents (PDF, DOCX, etc.) in the `AI-Assistant/data/raw/` directory.

2.  **Start the Assistant**:
    ```bash
    python AI-Assistant/main.py
    ```

3.  **Interaction**:
    - Ask questions about your materials.
    - Provide a YouTube URL to ask questions about a specific video.
    - Type `exit` or `quit` to stop.

## 🐳 Docker Support

You can also run the project using Docker:

```bash
docker-compose up --build
```

## 📁 Project Structure

```
.
├── AI-Assistant/
│   ├── config/          # Configuration settings (paths, models, etc.)
│   ├── data/            # Course materials and vector database
│   ├── src/             # Core logic (RAG pipeline, processors, retriever)
│   ├── utils/           # Helper functions
│   ├── tests/           # Verification scripts
│   └── main.py          # Entry point
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
└── requirements.txt     # Python dependencies
```

## ⚙️ Configuration

Settings can be adjusted in `AI-Assistant/config/settings.py`, including:
- `CHUNK_SIZE` and `CHUNK_OVERLAP` for text splitting.
- `EMBEDDING_MODEL` for vectorization.
- `OLLAMA_MODEL` to change the LLM.
- `ENABLE_OCR` to toggle image processing.
