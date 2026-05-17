"""
Shared ingestion pipeline — used by both Agent 3 and Agent 6.

Provides:
    - process_pdf()     — Detectron2 layout detection + VLM figure description
    - upsert_corpus()   — Semantic chunking + ChromaDB upsert
    - rebuild_bm25()    — Full BM25 index rebuild from the collection
"""

import os
import re
import pickle

import cv2
import fitz
import numpy as np
import ollama
import chromadb
from rank_bm25 import BM25Okapi
import layoutparser as lp

from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from config import (
    VECTORDB_PATH,
    COLLECTION_NAME,
    BM25_INDEX_PATH,
    DETECTRON_WEIGHTS,
    DETECTRON_CONFIG,
    DETECTRON_LABEL_MAP,
    DETECTRON_SCORE_THRESH,
    IMAGES_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    PDF_RENDER_DPI,
    CHUNK_MIN_LENGTH,
    EMBED_BATCH_SIZE,
    EMBED_MAX_CHARS,
    SEMANTIC_CHUNKER_TYPE,
    SEMANTIC_CHUNKER_AMOUNT,
)
from shared.log import get_logger
from shared.retry import retry
from shared.db import get_max_chunk_index

logger = get_logger("ingestion")


# ─── Text helpers ─────────────────────────────────────────────────────────────


def find_caption(text_blocks, bbox, box_type):
    """Find the nearest caption text below a figure/table bounding box."""
    x1, y1, x2, y2 = bbox
    best, min_dist = "", float("inf")
    for b in text_blocks:
        tx0, ty0, tx1, ty1 = b["bbox"]
        text = b["text"]
        horiz_overlap = max(0, min(x2, tx1) - max(x1, tx0))
        if horiz_overlap < 10 and (x2 - x1) > 100:
            continue
        if box_type == "Figure":
            dist = ty0 - y2
            if -20 < dist < 400:
                score = dist - (200 if text.lower().startswith("fig") else 0)
                if score < min_dist:
                    min_dist, best = score, text
    return best


def clean_text(text):
    """Remove fragmented single-character spacing artifacts from PDF extraction."""
    text = re.sub(r'\b(\w) (\w) ', r'\1\2', text)
    for _ in range(20):
        text = re.sub(r'(?<!\w)(\w) (\w)(?!\w)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── VLM call (with retry) ───────────────────────────────────────────────────


@retry(max_retries=3, backoff=2.0)
def describe_figure(image_path: str, fig_type: str, context: str) -> str:
    """Ask the VLM to describe a cropped figure/table image."""
    prompt = (
        f"You are analysing scientific plots. Describe this {fig_type.lower()}. "
        f"Extract textual information, data and trends.\n\n"
        f"Surrounding Document Context:\n{context}. Answer in 3-5 sentences at max."
    )
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_path]}],
    )
    try:
        return response.message.content
    except AttributeError:
        return response["message"]["content"]


# ─── Core PDF processing ─────────────────────────────────────────────────────


def process_pdf(pdf_path: str, citation_string: str, detectron_weights=None, images_dir=None):
    """
    Run Detectron2 layout detection and VLM figure description on a single PDF.

    Args:
        pdf_path:          Path to the PDF file.
        citation_string:   Citation label for metadata tagging.
        detectron_weights: Path to the Detectron2 checkpoint (defaults to config).
        images_dir:        Directory to save cropped figures (defaults to config).

    Returns:
        list[dict]: Corpus entries (text blocks + figure descriptions).
    """
    detectron_weights = detectron_weights or DETECTRON_WEIGHTS
    images_dir = images_dir or IMAGES_DIR
    os.makedirs(images_dir, exist_ok=True)

    logger.info("Processing pages for %s…", pdf_path)
    corpus = []
    dpi = PDF_RENDER_DPI
    zoom = dpi / 72.0

    # Init model inside worker for CUDA compatibility
    model = lp.Detectron2LayoutModel(
        config_path=DETECTRON_CONFIG,
        model_path=detectron_weights,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", DETECTRON_SCORE_THRESH],
        label_map=DETECTRON_LABEL_MAP,
    )

    try:
        pdf = fitz.open(pdf_path)
        doc_name = os.path.basename(pdf_path)
        pdf_name = doc_name.strip().replace(" ", "_").lower()

        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)

            layout = model.detect(img)
            raw_blocks = page.get_text("blocks")
            text_blocks_img = []

            for b in raw_blocks:
                if b[6] == 0:
                    tx1, ty1, tx2, ty2 = [c * zoom for c in b[:4]]
                    text = b[4].replace("\n", " ")
                    text_blocks_img.append({"bbox": (tx1, ty1, tx2, ty2), "text": text.strip()})
                    if len(text) > 4:
                        corpus.append({
                            "document": pdf_name,
                            "citation": citation_string,
                            "page": page_idx,
                            "type": "text",
                            "content": text,
                        })

            for i, fig in enumerate(b for b in layout if b.type in ["Figure", "Table"]):
                pad = 20
                x1, y1, x2, y2 = fig.coordinates
                x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
                x2, y2 = min(img.shape[1], int(x2 + pad)), min(img.shape[0], int(y2 + pad))

                cropped = img[y1:y2, x1:x2]
                out_name = f"{pdf_name}_p{page_idx}_f{i}.png"
                out_path = os.path.join(images_dir, out_name)
                cv2.imwrite(out_path, cropped)

                context = find_caption(text_blocks_img, (x1, y1, x2, y2), fig.type)

                try:
                    vlm_desc = describe_figure(out_path, fig.type, context)
                except Exception as e:
                    logger.error("VLM failed on %s: %s", out_name, e)
                    vlm_desc = "Description generation failed."

                corpus.append({
                    "document": pdf_name,
                    "citation": citation_string,
                    "page": page_idx,
                    "type": fig.type.lower(),
                    "content": vlm_desc,
                    "metadata": {"image_path": out_name, "caption": context},
                })

    except Exception as e:
        logger.error("Failed to process %s: %s", pdf_path, e)

    return corpus


# ─── ChromaDB upsert ─────────────────────────────────────────────────────────


def upsert_corpus(corpus: list[dict]):
    """
    Semantic-chunk the corpus and upsert into ChromaDB.

    Returns the number of new chunks inserted.
    """
    logger.info("Initializing chunker and embedding model…")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=SEMANTIC_CHUNKER_TYPE,
        breakpoint_threshold_amount=SEMANTIC_CHUNKER_AMOUNT,
    )
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Chunk text entries, pass figures/tables through directly
    total_corpus = []
    for entry in corpus:
        if entry["type"] in ["figure", "table"]:
            total_corpus.append(entry)
        elif entry["type"] == "text":
            content = clean_text(entry["content"])
            if len(content) < CHUNK_MIN_LENGTH:
                continue
            try:
                for j, doc in enumerate(chunker.create_documents([content])):
                    total_corpus.append({
                        "document": entry["document"],
                        "citation": entry["citation"],
                        "page": entry["page"],
                        "type": "text_chunk",
                        "content": doc.page_content,
                        "metadata": {"original_text": content, "chunk_index": j},
                    })
            except Exception as e:
                logger.error("Chunker failed: %s", e)

    # Deduplicate and prepare for insertion
    current_index = get_max_chunk_index(collection)
    documents, metadatas, ids, seen = [], [], [], set()

    for entry in total_corpus:
        content = entry["content"].strip()
        if content in seen or len(content) < CHUNK_MIN_LENGTH:
            continue
        seen.add(content)
        meta = {
            "document": entry["document"],
            "page": entry["page"],
            "type": entry["type"],
            "citation_source": entry["citation"],
        }
        if "metadata" in entry:
            for k, v in entry["metadata"].items():
                meta[f"extra_{k}"] = str(v)
        documents.append(content)
        metadatas.append(meta)
        ids.append(f"chunk_{current_index}")
        current_index += 1

    if documents:
        logger.info("Embedding and ingesting %d chunks…", len(documents))
        for i in range(0, len(documents), EMBED_BATCH_SIZE):
            b_docs = [d[:EMBED_MAX_CHARS] for d in documents[i : i + EMBED_BATCH_SIZE]]
            collection.add(
                embeddings=embeddings.embed_documents(b_docs),
                documents=b_docs,
                metadatas=metadatas[i : i + EMBED_BATCH_SIZE],
                ids=ids[i : i + EMBED_BATCH_SIZE],
            )
        logger.info("✓ Ingested %d chunks into ChromaDB.", len(documents))
    else:
        logger.info("No new chunks to insert.")

    return len(documents)


# ─── BM25 rebuild ─────────────────────────────────────────────────────────────


def rebuild_bm25():
    """Rebuild the BM25 index from the entire ChromaDB collection."""
    logger.info("Rebuilding BM25 index…")
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    paired, limit, offset = [], 1000, 0
    while True:
        batch = collection.get(include=["documents"], limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for doc, cid in zip(batch["documents"], batch["ids"]):
            try:
                paired.append((int(cid.split("_")[1]), doc))
            except (ValueError, IndexError):
                pass
        offset += limit

    paired.sort(key=lambda x: x[0])
    texts = [p[1] for p in paired]
    bm25 = BM25Okapi([re.findall(r'\w+', t.lower()) for t in texts])

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("✓ BM25 index rebuilt (%d documents).", len(texts))
