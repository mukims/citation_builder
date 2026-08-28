"""
Shared hybrid search implementation.

Used by Agent 4 (interactive), Agent 5 (batch), and evaluate_rag.py.

The embeddings model is accepted as a parameter so callers can reuse a
singleton instance across many calls (critical for Agent 5's per-sentence loop).
"""

import re

import numpy as np

from config import EMBED_MODEL, RRF_K, DEFAULT_TOP_K
from shared.log import get_logger

logger = get_logger("search")

# Module-level singleton — created once, reused by all callers that don't
# pass their own instance.
_embeddings_singleton = None


def _get_embeddings():
    global _embeddings_singleton
    if _embeddings_singleton is None:
        # Imported here so callers that pass their own embeddings model — and
        # tests that pass neither — do not need the LangChain stack installed.
        from langchain_ollama import OllamaEmbeddings

        _embeddings_singleton = OllamaEmbeddings(model=EMBED_MODEL)
    return _embeddings_singleton


def hybrid_search(
    query: str,
    collection,
    bm25,
    texts: list[str],
    metadatas: list[dict],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = RRF_K,
    embeddings_model=None,
) -> list[dict]:
    """
    Perform hybrid (BM25 sparse + dense vector) search with Reciprocal Rank Fusion.

    Args:
        query:            The search query string.
        collection:       ChromaDB collection handle.
        bm25:             BM25Okapi instance.
        texts:            Ordered list of all document texts (aligned with BM25 index).
        metadatas:        Ordered list of metadata dicts (aligned with texts).
        top_k:            Number of final results to return.
        rrf_k:            RRF fusion constant (higher = less rank bias).
        embeddings_model: Optional pre-initialised OllamaEmbeddings instance.
                          If None, a module-level singleton is used.

    Returns:
        list[dict]: Each entry has keys: chunk_index, text, metadata, rrf_score.
    """
    if embeddings_model is None:
        embeddings_model = _get_embeddings()

    k_cand = max(15, top_k * 3)

    # 1. Sparse (BM25) retrieval
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Only the top k_cand matter, so partition instead of sorting the whole
    # corpus. Agent 5 calls this once per sentence needing a citation, and a
    # full argsort is O(n log n) over every chunk in the database each time.
    if k_cand < len(bm25_scores):
        top_unordered = np.argpartition(bm25_scores, -k_cand)[-k_cand:]
        sparse_top_indices = top_unordered[np.argsort(bm25_scores[top_unordered])[::-1]]
    else:
        sparse_top_indices = np.argsort(bm25_scores)[::-1]

    # 2. Dense (embedding) retrieval
    query_emb = embeddings_model.embed_query(query)
    dense_results = collection.query(
        query_embeddings=[query_emb],
        n_results=k_cand,
        include=["documents", "metadatas", "distances"],
    )

    # The integer in a "chunk_N" id doubles as the position of that chunk in
    # `texts`, because both the BM25 corpus and `texts` are built by sorting the
    # collection on that same integer. That holds only while the ids are
    # contiguous from zero, which is true by construction (they are assigned
    # sequentially and never deleted) but is not enforced anywhere. Anything out
    # of range is dropped rather than silently indexing the wrong chunk.
    dense_ids_ordered = []
    if dense_results["ids"] and dense_results["ids"][0]:
        for id_str in dense_results["ids"][0]:
            try:
                idx = int(id_str.split("_")[1])
            except (ValueError, IndexError):
                continue
            if 0 <= idx < len(texts):
                dense_ids_ordered.append(idx)
            else:
                logger.warning(
                    "Dense hit %s is outside the %d loaded chunks — the id space "
                    "and the BM25 ordering have diverged; rebuild the index.",
                    id_str, len(texts),
                )

    # 3. Reciprocal Rank Fusion
    fused_scores = {}
    for rank, idx in enumerate(sparse_top_indices):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rank + 1 + rrf_k))
    for rank, idx in enumerate(dense_ids_ordered):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rank + 1 + rrf_k))

    # 4. Rank and return top_k
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, rrf_score in ranked:
        if idx < len(texts):
            results.append({
                # Cast: sparse indices arrive as numpy integers and dense ones
                # as Python ints, so the field's type depended on which
                # retriever surfaced the chunk.
                "chunk_index": int(idx),
                "text": texts[idx],
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "rrf_score": round(rrf_score, 5),
            })

    return results
