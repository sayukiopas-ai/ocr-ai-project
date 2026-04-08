# 📚 OCR Knowledge Base

Thai/English document knowledge base with OCR, Hybrid Search, and RAG — powered by Typhoon Vision + Cohere + Qdrant.

## ✨ Features

| Feature | Tech |
|---------|------|
| **OCR** | Typhoon Vision API (Thai/English) |
| **Embedding** | Cohere Embed v3 (multilingual) |
| **Vector Search** | Qdrant |
| **Keyword Search** | BM25 (rank-bm25) |
| **Re-ranking** | Cohere Rerank |
| **RAG** | Typhoon LLM (OpenAI-compatible) |
| **UI** | Streamlit (multi-page) |

## 🏗️ Architecture

```
Upload → Parse → OCR → Chunk → Embed → Qdrant
                                          ↓
Query → Embed → Vector Search ─┬─→ Merge → Re-rank → LLM → Answer
                  BM25 Search ─┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- API Keys: Typhoon, Cohere

### 2. Setup

```bash
# Clone & enter
cd ocr_project

# Python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with your API keys

# Start Qdrant
docker compose up -d
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`

## 📁 Project Structure

```
ocr_project/
├── app.py                    # Main Streamlit app
├── config.py                 # Configuration & env vars
├── requirements.txt
├── docker-compose.yml        # Qdrant container
├── .env.example
├── modules/
│   ├── ocr.py               # Typhoon Vision OCR
│   ├── file_parser.py        # PDF/DOCX/XLSX text extraction
│   ├── chunker.py            # Paragraph-aware text splitting
│   ├── embedder.py           # Cohere Embed v3
│   ├── vector_store.py       # Qdrant CRUD
│   ├── bm25_search.py        # BM25 keyword search
│   ├── hybrid_search.py      # Vector + BM25 + Re-rank
│   ├── reranker.py           # Cohere Re-rank
│   └── rag.py                # RAG pipeline (streaming)
├── pages/
│   ├── 1_📄_Upload.py        # Upload & process documents
│   ├── 2_🔍_Search.py        # Hybrid search
│   └── 3_💬_Ask.py           # RAG Q&A chat
└── styles/
    └── custom.css            # Dark glassmorphism theme
```

## 📄 Pages

- **Upload** — Drag & drop files → automated OCR + embedding pipeline
- **Search** — Hybrid search with relevance scores and source metadata
- **Ask** — Chat-style RAG with streaming answers and source citations

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `TYPHOON_API_KEY` | Typhoon Vision / LLM API key |
| `COHERE_API_KEY` | Cohere Embed + Rerank API key |
| `QDRANT_HOST` | Qdrant server host (default: `localhost`) |
| `QDRANT_PORT` | Qdrant server port (default: `6333`) |
# ocr-ai-project
