# 📚 OCR Knowledge Base

Thai/English document knowledge base with OCR, Hybrid Search, and RAG — powered by Typhoon OCR + Ollama + Qdrant.

## ✨ Features

| Feature | Tech |
|---------|------|
| **OCR** | Typhoon OCR 1.5 (Ollama) |
| **Embedding** | Nomic Embed Text (Ollama) |
| **Vector Search** | Qdrant |
| **Keyword Search** | BM25 (rank-bm25) |
| **Re-ranking** | Qwen3 (Ollama) |
| **RAG** | Qwen3 (Ollama) |
| **UI** | Streamlit (multi-page) |

## 🏗️ Architecture

```
Upload → Parse → OCR (Typhoon) → Chunk → Embed (Ollama) → Qdrant
                                                           ↓
Query → Embed → Vector Search ─┬─→ Merge → Re-rank (Ollama) → LLM (Ollama) → Answer
                  BM25 Search ─┘
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- [Ollama](https://ollama.com/) (Local AI Server)

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
# Edit .env if needed (default points to localhost Ollama/Qdrant)

# Pull AI Models
ollama pull scb10x/typhoon-ocr1.5-3b
ollama pull nomic-embed-text
ollama pull qwen3:1.7b

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
├── app.py                    # Home & Upload page
├── config.py                 # Configuration & env vars
├── requirements.txt
├── docker-compose.yml        # Qdrant container
├── .env.example
├── modules/
│   ├── ocr.py               # Typhoon Vision OCR via Ollama
│   ├── file_parser.py        # PDF/DOCX/XLSX text extraction
│   ├── chunker.py            # Paragraph-aware text splitting
│   ├── embedder.py           # Ollama Embed (Nomic)
│   ├── vector_store.py       # Qdrant CRUD
│   ├── bm25_search.py        # BM25 keyword search
│   ├── hybrid_search.py      # Vector + BM25 + Re-rank
│   ├── reranker.py           # Ollama Re-rank (Qwen3)
│   ├── sidebar.py            # Custom Streamlit sidebar
│   └── rag.py                # RAG pipeline (streaming via Qwen3)
├── pages/
│   ├── 2_🔍_Search.py        # Hybrid search interface
│   ├── 3_💬_Ask.py           # RAG Q&A chat interface
│   └── 4_📚_Library.py       # Document management (Library)
└── styles/
    └── custom.css            # Dark glassmorphism theme
```

## 📄 Pages

- **Home / Upload** — Drag & drop files → automated OCR + embedding pipeline
- **Search** — Hybrid search with relevance scores and source metadata
- **Ask** — Chat-style RAG with streaming answers and source citations
- **Library** — View all indexed documents and manage (delete) them

## 🔑 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OCR_MODEL` | OCR model name | `scb10x/typhoon-ocr1.5-3b:latest` |
| `LLM_MODEL` | RAG LLM model name | `qwen3:1.7b` |
| `EMBEDDING_MODEL` | Embedding model name | `nomic-embed-text` |
| `RERANKER_MODEL` | Reranker model name | `qwen3:1.7b` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
