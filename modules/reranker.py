"""
Re-ranker module — uses Ollama chat endpoint with qwen3-reranker model.

The qwen3-reranker model is prompted as a relevance grader:
it receives a query + document and returns a relevance score (0-10).
"""

import re
import requests

import config


def _score_document(query: str, document: str, timeout_s: int = 6) -> float:
    """
    Score a single document's relevance to a query using the reranker model.

    Returns a float score between 0.0 and 1.0.
    """
    prompt = (
        "You are a relevance grader. Given a query and a document, "
        "rate how relevant the document is to the query on a scale of 0-10. "
        "Reply with ONLY a single number, nothing else.\n\n"
        f"Query: {query}\n\n"
        f"Document: {document[:2000]}\n\n"
        "Relevance score (0-10):"
    )

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.RERANKER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"].strip()

        # Extract the first number from the response
        match = re.search(r"(\d+(?:\.\d+)?)", content)
        if match:
            score = float(match.group(1))
            return min(score / 10.0, 1.0)  # Normalize to 0-1
        return 0.0
    except Exception:
        return -1.0  # Sentinel: scoring failed (timeout / error)


def rerank(
    query: str,
    documents: list[dict],
    top_n: int = 5,
    max_docs_to_score: int = 4,
) -> list[dict]:
    """
    Re-rank documents by relevance to the query using Ollama reranker.

    Args:
        query: The search query.
        documents: List of dicts, each must have a 'text' key.
        top_n: Number of top results to return.

    Returns:
        Top-N documents sorted by rerank_score (descending),
        each dict gets a 'rerank_score' field added.
    """
    if not documents:
        return []

    # Cap reranking workload so Ask page remains responsive.
    candidates = sorted(
        documents,
        key=lambda d: d.get("score", 0),
        reverse=True,
    )[:max_docs_to_score]

    scored = []
    for doc in candidates:
        text = doc.get("text", "")
        score = _score_document(query, text)
        base_score = float(doc.get("score", 0))
        if score < 0:
            # Reranker failed (timeout/error) — use base_score as fallback
            final_score = base_score * 0.5
        else:
            # Reranker succeeded — use its score (0.0 = irrelevant is valid)
            final_score = score
        scored.append({**doc, "rerank_score": final_score})

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]
