"""
BM25 search — keyword-based search over stored document chunks.

Thai-aware: uses character n-gram tokenization for Thai text so that
substring queries like "พรบ" match "พระราชบัญญัติ" and "คอม" matches
"คอมพิวเตอร์" without a full Thai word-segmentation library.
"""

import re
from typing import Optional

from rank_bm25 import BM25Okapi

from modules import vector_store

# ── N-gram settings ──────────────────────────────────────
_THAI_NGRAM_MIN: int = 2
_THAI_NGRAM_MAX: int = 4


def _is_thai(char: str) -> bool:
    """Check if a character is in the Thai Unicode block."""
    return "\u0e00" <= char <= "\u0e7f"


def _thai_ngrams(text: str) -> list[str]:
    """
    Generate character n-grams from contiguous Thai runs.
    E.g. "คอมพิว" with ngram 2-4 → ["คอ", "อม", "มพ", "พิ", "คอม", "อมพ", "มพิ", "คอมพ", "อมพิ"]
    """
    grams: list[str] = []
    # Find contiguous Thai character sequences
    for m in re.finditer(r"[\u0e00-\u0e7f]+", text):
        run = m.group(0)
        # Strip Thai combining marks for length calculation but keep original
        base = re.sub(r"[ัิีึืุู็์ํ่้๊๋]", "", run)
        if len(base) < _THAI_NGRAM_MIN:
            grams.append(run.lower())
            continue
        for n in range(_THAI_NGRAM_MIN, min(_THAI_NGRAM_MAX + 1, len(base) + 1)):
            for i in range(len(base) - n + 1):
                grams.append(base[i : i + n].lower())
    return grams


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    - English/numbers: split on whitespace/punctuation (standard)
    - Thai: character n-grams (2-4) for substring matching
    - Also includes full Thai whitespace-separated words
    """
    text_lower = text.lower()
    tokens: list[str] = []

    # English + number tokens
    eng_tokens = re.findall(r"[a-z0-9]+", text_lower)
    tokens.extend(t for t in eng_tokens if len(t) >= 2)

    # Thai: whitespace-separated words (some OCR output has spaces)
    thai_words = re.findall(r"[\u0e00-\u0e7f]+", text_lower)
    tokens.extend(thai_words)

    # Thai: character n-grams for substring matching
    tokens.extend(_thai_ngrams(text_lower))

    return tokens


def _tokenize_query(query: str) -> list[str]:
    """
    Tokenize a search query.
    Simpler than document tokenization — each Thai word/fragment
    becomes n-grams, and English tokens are kept as-is.
    """
    q = query.lower().strip()
    tokens: list[str] = []

    # English + number tokens
    eng_tokens = re.findall(r"[a-z0-9]+", q)
    tokens.extend(t for t in eng_tokens if len(t) >= 2)

    # Thai fragments: use both the full fragment and n-grams
    thai_parts = re.findall(r"[\u0e00-\u0e7f]+", q)
    for part in thai_parts:
        # Strip combining marks for the "base" form
        base = re.sub(r"[ัิีึืุู็์ํ่้๊๋]", "", part)
        tokens.append(part)  # full fragment
        if len(base) >= _THAI_NGRAM_MIN:
            tokens.extend(_thai_ngrams(part))

    return tokens


def build_bm25_index() -> tuple[Optional[BM25Okapi], list[dict]]:
    """
    Build a BM25 index from all stored documents in Qdrant.

    Returns:
        Tuple of (BM25 index, list of document dicts with 'id' and 'text').
        Returns (None, []) if no documents exist.
    """
    docs = vector_store.get_all_texts()
    if not docs:
        return None, []

    # Tokenize each document's text + source filename for better matching
    tokenized = []
    for d in docs:
        text = d.get("text", "")
        source = d.get("source", "")
        combined = f"{source} {text}"
        tokenized.append(_tokenize(combined))

    index = BM25Okapi(tokenized)
    return index, docs


def bm25_search(
    query: str,
    limit: int = 20,
    index: Optional[BM25Okapi] = None,
    docs: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Search using BM25.

    Args:
        query: Search query string.
        limit: Max results to return.
        index: Pre-built BM25 index (builds new one if None).
        docs: Document list corresponding to the index.

    Returns:
        List of dicts with 'id', 'score', 'text', and metadata.
    """
    if index is None or docs is None:
        index, docs = build_bm25_index()

    if index is None or not docs:
        return []

    tokenized_query = _tokenize_query(query)
    if not tokenized_query:
        return []

    scores = index.get_scores(tokenized_query)

    # Pair scores with documents and sort
    scored_docs = [
        {**doc, "score": float(score)}
        for doc, score in zip(docs, scores)
        if score > 0
    ]

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:limit]
