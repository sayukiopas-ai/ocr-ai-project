"""
Embedding module — supports both Ollama embedding endpoints.

Batches large input lists to avoid timeouts and show progress.
"""

from __future__ import annotations

import requests

import config

# Cache the dimension after first detection
_embedding_dim: int | None = None

# Maximum texts per single /api/embed request.
# Keeps each HTTP call fast and avoids Ollama OOM on large batches.
_BATCH_SIZE: int = 32


def get_embedding_dimension() -> int:
    """
    Detect the embedding dimension by running a test embed.

    Returns:
        Integer dimension of the embedding vector.
    """
    global _embedding_dim
    if _embedding_dim is not None:
        return _embedding_dim

    test = embed_texts(["dimension test"])
    _embedding_dim = len(test[0])
    return _embedding_dim


def embed_texts(
    texts: list[str],
    *,
    progress_callback=None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts via Ollama.

    Args:
        texts: List of strings to embed.
        progress_callback: Optional callable(done, total) for progress reporting.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    base = config.OLLAMA_BASE_URL.rstrip("/")
    all_vectors: list[list[float]] = []
    total = len(texts)

    # Process in batches
    for start in range(0, total, _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]

        if progress_callback:
            progress_callback(min(start + len(batch), total), total)

        # Newer Ollama versions support batch embedding at /api/embed.
        response = requests.post(
            f"{base}/api/embed",
            json={
                "model": config.EMBEDDING_MODEL,
                "input": batch,
                "keep_alive": "5m",
            },
            timeout=300,
        )
        if response.ok:
            data = response.json()
            all_vectors.extend(data["embeddings"])
            continue

        # Backward compatibility for older Ollama versions using /api/embeddings.
        # That endpoint is single-input, so we call it once per text.
        if response.status_code == 404:
            for text in batch:
                legacy = requests.post(
                    f"{base}/api/embeddings",
                    json={
                        "model": config.EMBEDDING_MODEL,
                        "prompt": text,
                    },
                    timeout=300,
                )
                legacy.raise_for_status()
                all_vectors.append(legacy.json()["embedding"])
            continue

        response.raise_for_status()

    return all_vectors


def embed_query(text: str) -> list[float]:
    """
    Generate embedding for a single query text.

    Args:
        text: Query string.

    Returns:
        Embedding vector.
    """
    result = embed_texts([text])
    return result[0]
