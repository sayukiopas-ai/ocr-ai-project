"""
Hybrid Search — combines vector search + BM25 + re-ranking.
"""

import re

from modules import embedder, vector_store, bm25_search, reranker


def hybrid_search(
    query: str,
    vector_limit: int = 8,
    bm25_limit: int = 8,
    final_top_n: int = 5,
) -> list[dict]:
    """
    Perform hybrid search: vector + BM25, then re-rank.

    Pipeline:
        1. Embed query → Vector search (top-20)
        2. BM25 search (top-20)
        3. Merge & deduplicate
        4. Re-rank with Ollama → top-N

    Args:
        query: Search query string.
        vector_limit: Max results from vector search.
        bm25_limit: Max results from BM25 search.
        final_top_n: Final number of results after re-ranking.

    Returns:
        List of re-ranked document dicts.
    """
    # 0. Filename/source keyword search (helps direct file-name queries)
    source_results = _search_by_source_name(query, limit=vector_limit)

    # 1. Vector search
    query_vector = embedder.embed_query(query)
    vector_results = vector_store.vector_search(
        query_vector=query_vector,
        limit=vector_limit,
    )

    # 2. BM25 search
    bm25_results = bm25_search.bm25_search(
        query=query,
        limit=bm25_limit,
    )

    # 3. Merge & deduplicate
    merged = _merge_results(source_results, vector_results, bm25_results)

    if not merged:
        return []

    # 4. Re-rank
    reranked = reranker.rerank(
        query=query,
        documents=merged,
        top_n=final_top_n,
    )

    return reranked


def _merge_results(
    source_results: list[dict],
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """
    Merge and deduplicate results from vector and BM25 search.

    Uses text content as dedup key since IDs may differ
    between vector store scroll and search results.
    """
    seen_texts: set[str] = set()
    merged: list[dict] = []

    # Source-name matches first for explicit file queries.
    for doc in source_results:
        text = doc.get("text", "")
        text_key = text[:200]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            doc["search_source"] = "source"
            merged.append(doc)

    # Vector results first (generally higher quality)
    for doc in vector_results:
        text = doc.get("text", "")
        text_key = text[:200]  # Use first 200 chars as dedup key
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            doc["search_source"] = "vector"
            merged.append(doc)

    # Then BM25 results
    for doc in bm25_results:
        text = doc.get("text", "")
        text_key = text[:200]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            doc["search_source"] = "bm25"
            merged.append(doc)

    return merged


def _search_by_source_name(query: str, limit: int = 8) -> list[dict]:
    """
    Find chunks whose source filename appears related to the query.
    """
    docs = vector_store.get_all_texts()
    if not docs:
        return []

    q = query.lower().strip()
    if not q:
        return []

    tokens = re.findall(r"[a-z0-9._-]+", q)
    if not tokens:
        return []

    matches: list[dict] = []
    seen_ids: set[str] = set()
    for doc in docs:
        source = str(doc.get("source", "")).lower()
        if not source:
            continue

        # Strong match if the full query or any meaningful token is in filename.
        token_hit = any(len(t) >= 3 and t in source for t in tokens)
        if q in source or token_hit:
            doc_id = str(doc.get("id", ""))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            matched = {
                **doc,
                # Keep filename hits highly ranked before rerank.
                "score": max(float(doc.get("score", 0)), 0.95),
            }
            matches.append(matched)
            if len(matches) >= limit:
                break

    return matches
