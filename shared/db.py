"""
Shared database loading utilities.

Consolidates the repeated ChromaDB + BM25 loading boilerplate
used by Agents 4, 5, and evaluate_rag.py.

Usage:
    from shared.db import load_search_resources
    collection, bm25, texts, metadatas = load_search_resources()
"""

import pickle

import chromadb

from config import VECTORDB_PATH, COLLECTION_NAME, BM25_INDEX_PATH
from shared.log import get_logger

logger = get_logger("db")


def load_search_resources():
    """
    Connect to ChromaDB, fetch all chunks in index order, and load the
    BM25 pickle.

    Returns:
        tuple: (collection, bm25, texts, metadatas)
    """
    logger.info("Connecting to ChromaDB at %s…", VECTORDB_PATH)
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Paginate all chunks from the collection
    paired_data = []
    limit, offset = 1000, 0
    while True:
        batch = collection.get(
            include=["documents", "metadatas"], limit=limit, offset=offset
        )
        if not batch["ids"]:
            break
        for doc, meta, chunk_id in zip(
            batch["documents"], batch["metadatas"], batch["ids"]
        ):
            try:
                idx = int(chunk_id.split("_")[1])
                paired_data.append((idx, doc, meta))
            except (ValueError, IndexError):
                logger.warning("Skipping chunk with unexpected ID format: %s", chunk_id)
        offset += limit

    paired_data.sort(key=lambda x: x[0])
    texts = [item[1] for item in paired_data]
    metadatas = [item[2] for item in paired_data]

    logger.info("Loaded %d chunks from ChromaDB.", len(texts))

    logger.info("Loading BM25 index from %s…", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)

    return collection, bm25, texts, metadatas


def get_max_chunk_index(collection) -> int:
    """
    Read the current maximum chunk index from the collection.

    Returns the next available index (max + 1), or 0 if the collection is empty.
    """
    max_idx = -1
    limit, offset = 1000, 0
    while True:
        batch = collection.get(limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for cid in batch["ids"]:
            try:
                max_idx = max(max_idx, int(cid.split("_")[1]))
            except (ValueError, IndexError):
                pass
        offset += limit
    return max_idx + 1
