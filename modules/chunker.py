"""
Text chunker — splits extracted text into overlapping chunks for embedding.
"""

import config


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    Uses a simple sentence-aware splitter:
    1. Split by paragraphs / double newlines first
    2. Then group paragraphs into chunks of ~chunk_size chars
    3. Apply overlap between consecutive chunks

    Args:
        text: Raw text to chunk.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        List of text chunks.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If single paragraph exceeds chunk_size, split by sentences
        if para_len > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split long paragraph by sentences
            sentences = _split_sentences(para)
            for sent in sentences:
                if current_length + len(sent) > chunk_size and current_chunk:
                    chunks.append("\n".join(current_chunk))
                    # Keep overlap
                    overlap_text = "\n".join(current_chunk)
                    if len(overlap_text) > chunk_overlap:
                        overlap_text = overlap_text[-chunk_overlap:]
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)
                current_chunk.append(sent)
                current_length += len(sent)
        else:
            if current_length + para_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                # Keep overlap
                overlap_text = "\n".join(current_chunk)
                if len(overlap_text) > chunk_overlap:
                    overlap_text = overlap_text[-chunk_overlap:]
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)

            current_chunk.append(para)
            current_length += para_len

    # Flush remaining
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for Thai + English."""
    import re

    # Split on common sentence endings
    parts = re.split(r"(?<=[.!?。\n])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Chunk a list of page dicts (from OCR/PDF extraction).

    Args:
        pages: List of {"page": int, "text": str}

    Returns:
        List of {"chunk_id": int, "page": int, "text": str}
    """
    all_chunks: list[dict] = []
    chunk_id = 0

    for page_data in pages:
        page_num = page_data.get("page", 0)
        text = page_data.get("text", "")
        chunks = chunk_text(text, chunk_size, chunk_overlap)

        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": chunk,
            })
            chunk_id += 1

    return all_chunks
