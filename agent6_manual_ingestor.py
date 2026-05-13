"""
Agent 6 — Manual PDF Ingestor (agent6_manual_ingestor.py)

Monitors the `pulled_pdfs/` directory for manually dropped PDF files.
When a new PDF is detected, it processes the file directly through the full
multimodal ingestion pipeline (Detectron2 layout detection → gemma4 VLM figure
description → ChromaDB + BM25 indexing) without requiring a citation string from
downloaded.json.

This is the companion to Agent 3 for cases where Agent 2 could not automatically
find and download a paper (e.g., paywalled papers that you have downloaded manually).

Usage (standalone):
    python agent6_manual_ingestor.py                 # watch pulled_pdfs/ continuously
    python agent6_manual_ingestor.py --once <file>   # ingest a single PDF directly

The script can also be imported and called programmatically:
    from agent6_manual_ingestor import ingest_manual_pdf
    ingest_manual_pdf("pulled_pdfs/mypaper.pdf", citation_string="Smith et al. 2024")
"""

import os
import re
import sys
import time
import threading
import argparse
import pickle

import cv2
import fitz
import numpy as np
import ollama
import chromadb
from rank_bm25 import BM25Okapi
import layoutparser as lp

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Error: 'watchdog' is not installed. Run: pip install watchdog")
    sys.exit(1)

try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError as e:
    print(f"Error: LangChain modules not found. {e}")
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

PULLED_PDFS_DIR    = "pulled_pdfs"
VECTORDB_PATH      = "./physics_vectordb"
COLLECTION_NAME    = "physics_papers"
BM25_INDEX_PATH    = "./bm25_index.pkl"
DETECTRON_WEIGHTS  = os.path.abspath("../model_final.pth")
IMAGES_DIR         = "../extracted_data/images"
EMBED_MODEL        = "nomic-embed-text"
VLM_MODEL          = "gemma4:latest"
COOLDOWN_SECONDS   = 5   # debounce — wait for file write to finish before processing

# ─── Layout & Text Helpers ────────────────────────────────────────────────────

def find_cap(text_blocks, bbox, box_type):
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

# ─── Core Ingestion Logic ──────────────────────────────────────────────────────

def ingest_manual_pdf(pdf_path: str, citation_string: str | None = None, workers: int = 1):
    """
    Ingest a single manually placed PDF into the ChromaDB vector database.

    Args:
        pdf_path:        Absolute or relative path to the PDF file.
        citation_string: Optional citation/reference label to tag the document with.
                         If None, the PDF filename (without extension) is used.
        workers:         Unused here (kept for API compatibility), always single-file.
    """
    if not os.path.exists(pdf_path):
        print(f"[Agent 6] File not found: {pdf_path}")
        return

    if citation_string is None:
        citation_string = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"\n[Agent 6] Processing: {pdf_path}")
    print(f"[Agent 6] Citation label: '{citation_string}'")

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # ── Step 1: Detectron2 layout detection + VLM figure description ──────────
    model = lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
        model_path=DETECTRON_WEIGHTS,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )

    corpus = []
    dpi = 200
    zoom = dpi / 72.0

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
                            "document":  pdf_name,
                            "citation":  citation_string,
                            "page":      page_idx,
                            "type":      "text",
                            "content":   text,
                        })

            for i, fig in enumerate([b for b in layout if b.type in ["Figure", "Table"]]):
                pad = 20
                x1, y1, x2, y2 = fig.coordinates
                x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
                x2, y2 = min(img.shape[1], int(x2 + pad)), min(img.shape[0], int(y2 + pad))

                cropped = img[y1:y2, x1:x2]
                out_name = f"{pdf_name}_p{page_idx}_f{i}.png"
                out_path = os.path.join(IMAGES_DIR, out_name)
                cv2.imwrite(out_path, cropped)

                context = find_cap(text_blocks_img, (x1, y1, x2, y2), fig.type)

                try:
                    response = ollama.chat(
                        model=VLM_MODEL,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"You are analysing scientific plots. Describe this {fig.type.lower()}. "
                                f"Extract textual information, data and trends.\n\n"
                                f"Surrounding Document Context:\n{context}. Answer in 3-5 sentences at max."
                            ),
                            "images": [out_path],
                        }],
                    )
                    try:
                        vlm_desc = response.message.content
                    except AttributeError:
                        vlm_desc = response["message"]["content"]
                except Exception as e:
                    print(f"[Agent 6] VLM failed on {out_name}: {e}")
                    vlm_desc = "Description generation failed."

                corpus.append({
                    "document":  pdf_name,
                    "citation":  citation_string,
                    "page":      page_idx,
                    "type":      fig.type.lower(),
                    "content":   vlm_desc,
                    "metadata":  {"image_path": out_name, "caption": context},
                })

    except Exception as e:
        print(f"[Agent 6] Failed to process {pdf_path}: {e}")
        return

    if not corpus:
        print("[Agent 6] No content extracted. Aborting.")
        return

    # ── Step 2: Semantic chunking + ChromaDB upsert ───────────────────────────
    print("[Agent 6] Initializing chunker and embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chroma_client = chromadb.PersistentClient(path=VECTORDB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_corpus = []
    for entry in corpus:
        if entry["type"] in ["figure", "table"]:
            total_corpus.append(entry)
        elif entry["type"] == "text":
            content = clean_text(entry["content"])
            if len(content) < 10:
                continue
            try:
                for j, doc in enumerate(chunker.create_documents([content])):
                    total_corpus.append({
                        "document": entry["document"],
                        "citation": entry["citation"],
                        "page":     entry["page"],
                        "type":     "text_chunk",
                        "content":  doc.page_content,
                        "metadata": {"original_text": content, "chunk_index": j},
                    })
            except Exception as e:
                print(f"[Agent 6] Chunker error: {e}")

    # Find current max chunk index to continue sequentially
    max_idx, limit, offset = -1, 1000, 0
    while True:
        batch = collection.get(limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for cid in batch["ids"]:
            try:
                max_idx = max(max_idx, int(cid.split("_")[1]))
            except Exception:
                pass
        offset += limit
    current_index = max_idx + 1

    documents, metadatas, ids, seen = [], [], [], set()
    for entry in total_corpus:
        content = entry["content"].strip()
        if content in seen or len(content) < 10:
            continue
        seen.add(content)
        meta = {
            "document":       entry["document"],
            "page":           entry["page"],
            "type":           entry["type"],
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
        print(f"[Agent 6] Embedding and ingesting {len(documents)} chunks...")
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            b_docs = [d[:4000] for d in documents[i:i + batch_size]]
            collection.add(
                embeddings=embeddings.embed_documents(b_docs),
                documents=b_docs,
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
        print(f"[Agent 6] ✓ Ingested {len(documents)} chunks from '{pdf_path}'.")
    else:
        print("[Agent 6] No new chunks to insert.")
        return

    # ── Step 3: Rebuild BM25 index ────────────────────────────────────────────
    print("[Agent 6] Rebuilding BM25 index...")
    paired, limit, offset = [], 1000, 0
    while True:
        batch = collection.get(include=["documents"], limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for doc, cid in zip(batch["documents"], batch["ids"]):
            try:
                paired.append((int(cid.split("_")[1]), doc))
            except Exception:
                pass
        offset += limit

    paired.sort(key=lambda x: x[0])
    texts = [p[1] for p in paired]
    bm25 = BM25Okapi([re.findall(r'\w+', t.lower()) for t in texts])
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print("[Agent 6] ✓ BM25 index rebuilt successfully.\n")


# ─── Watchdog Handler ─────────────────────────────────────────────────────────

class ManualPDFHandler(FileSystemEventHandler):
    """Debounced watchdog handler for the pulled_pdfs/ directory."""

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        self._schedule(event)

    def on_moved(self, event):
        # Handles files moved/renamed into the directory
        self._schedule(event, use_dest=True)

    def _schedule(self, event, use_dest=False):
        path = getattr(event, "dest_path", None) if use_dest else event.src_path
        if not path or event.is_directory or not path.lower().endswith(".pdf"):
            return
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(COOLDOWN_SECONDS, self._process, args=[path])
            self._timers[path] = timer
            timer.start()
            print(f"[Agent 6 Watcher] PDF detected: {os.path.basename(path)} — processing in {COOLDOWN_SECONDS}s...")

    def _process(self, path):
        with self._lock:
            self._timers.pop(path, None)
        ingest_manual_pdf(path)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agent 6 — Manual PDF Ingestor. Watches pulled_pdfs/ for new files."
    )
    parser.add_argument(
        "--once",
        metavar="PDF_PATH",
        help="Ingest a single PDF immediately instead of watching the directory.",
    )
    parser.add_argument(
        "--citation",
        metavar="CITATION_STRING",
        default=None,
        help="Optional citation label for --once mode. Defaults to the PDF filename.",
    )
    args = parser.parse_args()

    if args.once:
        ingest_manual_pdf(args.once, citation_string=args.citation)
        return

    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)
    handler = ManualPDFHandler()
    observer = Observer()
    observer.schedule(handler, path=PULLED_PDFS_DIR, recursive=False)
    observer.start()

    print(f"[Agent 6] Watching '{PULLED_PDFS_DIR}/' for manually placed PDFs...")
    print("Drop any PDF into the folder to auto-ingest it into ChromaDB.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Agent 6] Stopping watcher...")
        with handler._lock:
            for t in handler._timers.values():
                t.cancel()
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
