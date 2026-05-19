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
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

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

# Pre-compiled regex patterns for clean_text (avoid recompilation per call)
_RE_SPACED_PAIR_BOUNDARY = re.compile(r'\b(\w) (\w) ')
_RE_SPACED_PAIR_STANDALONE = re.compile(r'(?<!\w)(\w) (\w)(?!\w)')
_RE_MULTI_SPACE = re.compile(r'\s+')


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
    """Remove fragmented single-character spacing artifacts from PDF extraction.

    Uses pre-compiled regexes and converges early instead of looping a
    fixed 20 times.
    """
    text = _RE_SPACED_PAIR_BOUNDARY.sub(r'\1\2', text)
    # Iterate until stable (converges in 2-5 passes for typical PDF artefacts)
    for _ in range(20):
        new_text = _RE_SPACED_PAIR_STANDALONE.sub(r'\1\2', text)
        if new_text == text:
            break
        text = new_text
    text = _RE_MULTI_SPACE.sub(' ', text)
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


# ─── Detectron2 model cache ──────────────────────────────────────────────────
# In sequential mode, avoids re-loading the ~400MB model for every PDF.

_detectron_model_cache = {}


def _get_detectron_model(weights_path):
    """Return a cached Detectron2 model (loaded once per unique weights path)."""
    if weights_path not in _detectron_model_cache:
        logger.debug("Loading Detectron2 model from %s…", weights_path)
        _detectron_model_cache[weights_path] = lp.Detectron2LayoutModel(
            config_path=DETECTRON_CONFIG,
            model_path=weights_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", DETECTRON_SCORE_THRESH],
            label_map=DETECTRON_LABEL_MAP,
        )
    return _detectron_model_cache[weights_path]


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

    t0 = time.perf_counter()
    logger.info("Processing pages for %s…", pdf_path)
    corpus = []
    dpi = PDF_RENDER_DPI
    zoom = dpi / 72.0

    # Use cached model in sequential mode; in multiprocessing workers the
    # cache is per-process so each worker loads once then reuses.
    model = _get_detectron_model(detectron_weights)

    try:
        pdf = fitz.open(pdf_path)
        doc_name = os.path.basename(pdf_path)
        pdf_name = doc_name.strip().replace(" ", "_").lower()
        num_pages = len(pdf)

        for page_idx in tqdm(range(num_pages), desc=f"Parsing {doc_name}", leave=False):
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

        elapsed = time.perf_counter() - t0
        logger.info(
            "✓ %s: %d pages, %d entries extracted in %.1fs (%.2fs/page)",
            pdf_name, num_pages, len(corpus), elapsed, elapsed / max(num_pages, 1),
        )

    except Exception as e:
        logger.error("Failed to process %s: %s", pdf_path, e)

    return corpus


# ─── ChromaDB upsert ─────────────────────────────────────────────────────────


def upsert_corpus(corpus: list[dict]):
    """
    Semantic-chunk the corpus and upsert into ChromaDB.

    Batches text blocks together before chunking to reduce the number of
    embedding calls made by SemanticChunker (one call per batch instead of
    one per text block).

    Returns the number of new chunks inserted.
    """
    if not corpus:
        logger.info("Empty corpus — nothing to ingest.")
        return 0

    t0 = time.perf_counter()
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

    # ── Group text entries by (document, page) and batch-chunk ────────────
    # Instead of calling chunker.create_documents() once per text block
    # (which triggers an embedding call each time), we concatenate all text
    # on the same page and chunk once.  Figures/tables pass through directly.
    total_corpus = []
    page_text_groups = {}  # key: (document, citation, page) → list[str]

    for entry in corpus:
        if entry["type"] in ["figure", "table"]:
            total_corpus.append(entry)
        elif entry["type"] == "text":
            content = clean_text(entry["content"])
            if len(content) < CHUNK_MIN_LENGTH:
                continue
            key = (entry["document"], entry["citation"], entry["page"])
            page_text_groups.setdefault(key, []).append(content)

    # Chunk each page's concatenated text in one call
    for (doc, cit, page), texts in page_text_groups.items():
        merged = "\n\n".join(texts)
        try:
            docs = chunker.create_documents([merged])
            for j, doc_chunk in enumerate(docs):
                total_corpus.append({
                    "document": doc,
                    "citation": cit,
                    "page": page,
                    "type": "text_chunk",
                    "content": doc_chunk.page_content,
                    "metadata": {"chunk_index": j},
                })
        except Exception as e:
            logger.error("Chunker failed on page %d of %s: %s", page, doc, e)

    chunk_time = time.perf_counter() - t0
    logger.info("Chunking completed in %.1fs (%d entries)", chunk_time, len(total_corpus))

    # ── Deduplicate against existing DB content ──────────────────────────
    # Fetch existing document texts to skip re-ingesting identical chunks.
    existing_docs = set()
    limit, offset = 5000, 0
    while True:
        batch = collection.get(include=["documents"], limit=limit, offset=offset)
        if not batch or not batch["documents"]:
            break
        existing_docs.update(batch["documents"])
        offset += limit

    current_index = get_max_chunk_index(collection)
    documents, metadatas, ids, seen = [], [], [], set()

    for entry in total_corpus:
        content = entry["content"].strip()
        if content in seen or content in existing_docs or len(content) < CHUNK_MIN_LENGTH:
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
        embed_t0 = time.perf_counter()
        for i in range(0, len(documents), EMBED_BATCH_SIZE):
            b_docs = [d[:EMBED_MAX_CHARS] for d in documents[i : i + EMBED_BATCH_SIZE]]
            collection.add(
                embeddings=embeddings.embed_documents(b_docs),
                documents=b_docs,
                metadatas=metadatas[i : i + EMBED_BATCH_SIZE],
                ids=ids[i : i + EMBED_BATCH_SIZE],
            )
        elapsed = time.perf_counter() - t0
        embed_elapsed = time.perf_counter() - embed_t0
        logger.info(
            "✓ Ingested %d chunks into ChromaDB in %.1fs (embed: %.1fs, total: %.1fs)",
            len(documents), elapsed, embed_elapsed, elapsed,
        )
    else:
        logger.info("No new chunks to insert (all duplicates or empty).")

    return len(documents)


# ─── Already-ingested check ──────────────────────────────────────────────────


def get_ingested_documents(collection=None) -> set[str]:
    """Return the set of document filenames already present or processed.
    
    Checks both ChromaDB and a local manifest file to ensure we don't
    re-process duplicates or empty PDFs over and over.
    """
    docs = set()
    manifest_path = os.path.join(VECTORDB_PATH, "ingestion_manifest.txt")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    docs.add(line.strip())

    if collection is None:
        chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
        try:
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            return docs

    limit, offset = 5000, 0
    while True:
        batch = collection.get(include=["metadatas"], limit=limit, offset=offset)
        if not batch or not batch["metadatas"]:
            break
        for meta in batch["metadatas"]:
            if meta and "document" in meta:
                docs.add(meta["document"])
        offset += limit
        
    return docs

def mark_document_ingested(pdf_name: str):
    """Mark a document as processed so it's not re-ingested."""
    os.makedirs(VECTORDB_PATH, exist_ok=True)
    manifest_path = os.path.join(VECTORDB_PATH, "ingestion_manifest.txt")
    with open(manifest_path, "a") as f:
        f.write(pdf_name + "\n")


# ─── BM25 rebuild ─────────────────────────────────────────────────────────────


def rebuild_bm25():
    """Rebuild the BM25 index from the entire ChromaDB collection."""
    t0 = time.perf_counter()
    logger.info("Rebuilding BM25 index…")
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Use larger page size to reduce round-trips
    paired, limit, offset = [], 5000, 0
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
    elapsed = time.perf_counter() - t0
    logger.info("✓ BM25 index rebuilt (%d documents) in %.1fs.", len(texts), elapsed)
