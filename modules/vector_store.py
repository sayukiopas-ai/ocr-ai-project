"""
Qdrant vector store — handles collection management and CRUD operations.
"""

import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

import config
from modules.embedder import get_embedding_dimension

_client: QdrantClient | None = None

# Cache for expensive scroll() used by BM25 + filename search.
_all_texts_cache: list[dict] | None = None
_all_texts_cache_at: float | None = None
_ALL_TEXTS_CACHE_TTL_S: float = 60.0


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return _client


def ensure_collection() -> None:
    """Create the collection if it doesn't exist, or recreate if dimension changed."""
    client = _get_client()
    expected_dim = get_embedding_dimension()
    collections = [c.name for c in client.get_collections().collections]

    if config.QDRANT_COLLECTION in collections:
        # Check if vector dimension matches the current embedding model
        info = client.get_collection(config.QDRANT_COLLECTION)
        current_dim = info.config.params.vectors.size
        if current_dim != expected_dim:
            import logging
            logging.warning(
                "Qdrant collection '%s' has dimension %d but embedding model "
                "produces %d. Recreating collection...",
                config.QDRANT_COLLECTION, current_dim, expected_dim,
            )
            client.delete_collection(config.QDRANT_COLLECTION)
            collections.remove(config.QDRANT_COLLECTION)

    if config.QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=expected_dim,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """
    Upsert chunks with their embeddings into Qdrant.

    Args:
        chunks: List of {"chunk_id": int, "page": int, "text": str}
        embeddings: Corresponding embedding vectors.
        metadata: Extra metadata to attach (e.g., source filename).

    Returns:
        List of generated point IDs.
    """
    client = _get_client()
    ensure_collection()

    points = []
    point_ids = []

    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        payload = {
            "text": chunk["text"],
            "page": chunk.get("page", 0),
            "chunk_id": chunk.get("chunk_id", 0),
        }
        if metadata:
            payload.update(metadata)

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    # Batch upsert (Qdrant handles batching internally)
    client.upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=points,
    )

    invalidate_cache()
    return point_ids


def vector_search(
    query_vector: list[float],
    limit: int = 20,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Search Qdrant by vector similarity.

    Returns list of dicts with 'id', 'score', 'text', and metadata.
    """
    ensure_collection()
    client = _get_client()

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            **hit.payload,
        }
        for hit in results.points
    ]


def get_all_texts() -> list[dict]:
    """
    Retrieve all stored texts (for BM25 index building).

    Returns list of dicts with 'id' and 'text'.
    """
    global _all_texts_cache, _all_texts_cache_at
    now = time.time()
    if (
        _all_texts_cache is not None
        and _all_texts_cache_at is not None
        and (now - _all_texts_cache_at) < _ALL_TEXTS_CACHE_TTL_S
    ):
        return _all_texts_cache

    ensure_collection()
    client = _get_client()

    # Scroll through all points
    all_points = []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(results)
        if offset is None:
            break

    texts = [
        {"id": str(p.id), "text": p.payload.get("text", ""), **p.payload}
        for p in all_points
    ]
    _all_texts_cache = texts
    _all_texts_cache_at = now
    return texts


def delete_by_source(source: str) -> int:
    """
    Delete all points belonging to a specific source file.

    Only deletes chunks where payload.source == source.
    Other documents in the collection are NOT affected.

    Returns:
        Number of chunks that were deleted.
    """
    import logging

    ensure_collection()
    client = _get_client()

    # Count how many chunks belong to this source before deleting
    before_count = count_by_source(source)
    logging.info(
        "Deleting %d chunks for source '%s' from collection '%s'",
        before_count, source, config.QDRANT_COLLECTION,
    )

    if before_count == 0:
        logging.info("No chunks found for source '%s', nothing to delete.", source)
        invalidate_cache()
        return 0

    # Delete only points matching this specific source
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )

    invalidate_cache()

    # Verify deletion
    after_count = count_by_source(source)
    deleted = before_count - after_count
    logging.info(
        "Deleted %d chunks for source '%s'. Remaining: %d",
        deleted, source, after_count,
    )
    return deleted


def invalidate_cache() -> None:
    """Invalidate cached scroll results after write operations."""
    global _all_texts_cache, _all_texts_cache_at
    _all_texts_cache = None
    _all_texts_cache_at = None


def get_collection_info() -> dict:
    """Get collection stats."""
    client = _get_client()
    try:
        info = client.get_collection(config.QDRANT_COLLECTION)
        return {
            "points_count": info.points_count,
            "vectors_count": getattr(info, "indexed_vectors_count", 0),
            "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        }
    except Exception:
        return {"points_count": 0, "vectors_count": 0, "status": "not_found"}


def get_sources_stats() -> dict[str, int]:
    """
    Get chunk counts grouped by source filename from Qdrant.

    Returns:
        Dict mapping source filename -> number of chunks.
    """
    try:
        all_docs = get_all_texts()
    except Exception:
        return {}

    counts: dict[str, int] = {}
    for doc in all_docs:
        source = doc.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def count_by_source(source: str) -> int:
    """Count how many points belong to a specific source file."""
    try:
        ensure_collection()
        client = _get_client()
        result = client.count(
            collection_name=config.QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            exact=True,
        )
        return result.count
    except Exception:
        return 0

