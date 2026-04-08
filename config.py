"""
Central configuration — loads from .env and provides typed access.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Ollama ───────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model names (must be pulled in Ollama first)
# Typhoon OCR ~10 GiB RAM; if Ollama returns insufficient memory, set OCR_MODEL=moondream (or llava-phi3) in .env
OCR_MODEL: str = os.getenv("OCR_MODEL", "scb10x/typhoon-ocr1.5-3b:latest")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3:1.7b")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "qwen3:1.7b")

# ── Qdrant ───────────────────────────────────────────────
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Paths ────────────────────────────────────────────────
UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
