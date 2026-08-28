"""
Central configuration for the Citation Agent pipeline.

All hardcoded model names, paths, and tunable constants live here so that
changing a model or path only requires editing one file.
"""

import os

# ─── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Ollama Models ────────────────────────────────────────────────────────────
LLM_MODEL       = "gemma4:latest"
CHAT_MODEL      = "qwen2.5:7b"          # Lighter model for interactive research chat
EMBED_MODEL     = "nomic-embed-text"
EVAL_MODEL      = "deepseek-r1:14b"

# ─── Chat Model Runtime Options ──────────────────────────────────────────────
# Flash Attention + 8-bit quantized KV cache for lower latency on CPU
CHAT_OLLAMA_OPTIONS = {
    "num_ctx":    4096,        # Smaller context window = faster inference
    "num_thread": 16,          # Use most of the available CPU threads
    "flash_attn": True,        # Enable Flash Attention
    "kv_cache_type": "q8_0",   # 8-bit quantized KV cache
}

# ─── Vector Database ──────────────────────────────────────────────────────────
VECTORDB_PATH    = os.path.join(PROJECT_ROOT, "physics_vectordb")
COLLECTION_NAME  = "physics_papers"
BM25_INDEX_PATH  = os.path.join(PROJECT_ROOT, "bm25_index.pkl")

# ─── Directories ──────────────────────────────────────────────────────────────
RAW_DIR          = os.path.join(PROJECT_ROOT, "raw")
DRAFTS_DIR       = os.path.join(PROJECT_ROOT, "drafts")
PULLED_PDFS_DIR  = os.path.join(PROJECT_ROOT, "pulled_pdfs")
# Figure/table crops written during ingestion. Each crop is handed straight to
# the VLM and never read back — only the generated description enters the
# corpus — so this is a debugging artefact, not corpus data, and is safe to
# delete between runs. It previously resolved to ../extracted_data/images, a
# sibling of the project, which put the output outside the repo, outside
# version control and outside any backup taken of it.
IMAGES_DIR       = os.environ.get(
    "CITATION_IMAGES_DIR", os.path.join(PROJECT_ROOT, "images")
)

# ─── Data Files ───────────────────────────────────────────────────────────────
EXTRACTED_CITATIONS_PATH = os.path.join(PROJECT_ROOT, "extracted_citations.json")
DOWNLOADED_JSON_PATH     = os.path.join(PROJECT_ROOT, "downloaded.json")
FAILED_DOWNLOADS_PATH    = os.path.join(PROJECT_ROOT, "failed_downloads.json")

# ─── Evaluation (evaluate_rag.py) ─────────────────────────────────────────────
# Anchored to the project root: these were bare relative paths, so the
# evaluation only worked when run from this directory and wrote its output
# wherever it happened to be invoked from.
SAMPLE_INPUTS_PATH = os.path.join(PROJECT_ROOT, "sample_inputs")
EVAL_RESULTS_PATH  = os.path.join(PROJECT_ROOT, "evaluation_results.csv")

# ─── Detectron2 ───────────────────────────────────────────────────────────────
DETECTRON_WEIGHTS = os.path.join(PROJECT_ROOT, "model_final.pth")
DETECTRON_CONFIG  = "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
DETECTRON_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
DETECTRON_SCORE_THRESH = 0.5

# ─── Ingestion Tunables ───────────────────────────────────────────────────────
CHUNK_MIN_LENGTH         = 10       # Discard text chunks shorter than this
EMBED_BATCH_SIZE         = 1000     # ChromaDB upsert batch size
EMBED_MAX_CHARS          = 4000     # Truncate documents to this length before embedding
SEMANTIC_CHUNKER_TYPE    = "percentile"
SEMANTIC_CHUNKER_AMOUNT  = 90       # 90th percentile breakpoint

# ─── Search Tunables ──────────────────────────────────────────────────────────
RRF_K            = 60               # Reciprocal Rank Fusion constant
DEFAULT_TOP_K    = 3                # Default number of results to return

# ─── Orchestrator Tunables ────────────────────────────────────────────────────
PDF_COOLDOWN_SECONDS     = 30
DRAFT_COOLDOWN_SECONDS   = 2
MANUAL_COOLDOWN_SECONDS  = 5
DEFAULT_WORKERS          = 1

# ─── Agent 2 — Fetcher ───────────────────────────────────────────────────────
# Unpaywall requires a contact email on every request. Set UNPAYWALL_EMAIL in
# your environment; the placeholder below is only a fallback so the pipeline
# does not silently send someone else's address.
UNPAYWALL_EMAIL    = os.environ.get("UNPAYWALL_EMAIL", "your-email@example.com")
MAX_CITATION_LEN   = 500   # Skip citations longer than this (likely malformed)
ARXIV_RATE_LIMIT   = 3     # Seconds between arXiv requests
UNPAYWALL_SLEEP    = 0.5   # Courtesy sleep after Unpaywall downloads

# ─── Rendering / DPI ─────────────────────────────────────────────────────────
PDF_RENDER_DPI = 72
