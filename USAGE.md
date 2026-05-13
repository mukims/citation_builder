# Citation Agent — Detailed Usage Guide

This document is the single authoritative reference for running the multi-agent RAG citation pipeline. Read it top-to-bottom before executing any script.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Directory Structure](#3-directory-structure)
4. [Data Flow Diagram](#4-data-flow-diagram)
5. [Running the Full Pipeline (Automated — Recommended)](#5-running-the-full-pipeline-automated--recommended)
6. [Running Agents Manually (Step-by-Step)](#6-running-agents-manually-step-by-step)
   - [Agent 1 — Extractor](#agent-1--extractor-agent1_extractorpy)
   - [Agent 2 — Fetcher](#agent-2--fetcher-agent2_fetcherpy)
   - [Agent 3 — Ingestor](#agent-3--ingestor-agent3_ingestorpy)
   - [Agent 4 — Interactive Assistant](#agent-4--interactive-assistant-agent4_assistantpy)
   - [Agent 5 — Batch Citer](#agent-5--batch-citer-agent5_batch_citerpy)
7. [LangGraph Supervisor Agent](#7-langgraph-supervisor-agent)
8. [RAG Evaluation (Ragas)](#8-rag-evaluation-ragas)
9. [Output Files Reference](#9-output-files-reference)
10. [Configuration Reference](#10-configuration-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Overview

The pipeline automates the entire lifecycle from raw scientific PDFs → cited LaTeX drafts using a chain of five specialised agents, coordinated by a LangGraph-based supervisor:

| Agent | Script | Role |
|-------|--------|------|
| 1 | `agent1_extractor.py` | Parse reference strings out of source PDFs |
| 2 | `agent2_fetcher.py` | Download the referenced papers (open-access) |
| 3 | `agent3_ingestor.py` | Multimodal ingestion into ChromaDB + BM25 |
| 4 | `agent4_assistant.py` | Interactive single-sentence citation helper |
| 5 | `agent5_batch_citer.py` | Automated full-draft batch citation |
| — | `agent_graph.py` | **LangGraph supervisor** — LLM-driven tool-calling agent that decides which tools (agents) to invoke |
| — | `master_orchestrator.py` | Watchdog daemon — monitors directories and delegates events to the LangGraph supervisor |
| — | `evaluate_rag.py` | RAG evaluation using the Ragas framework |

**Local models used (via Ollama):**
- `gemma4:latest` — vision-language model for figure description (Agent 3), citation judgement (Agent 5), response generation (Agents 4 & 5), and supervisor reasoning (Agent Graph).
- `nomic-embed-text` — text embedding model used for dense vector search (Agents 3, 4, 5).
- `deepseek-r1:14b` — evaluator LLM used by the Ragas evaluation script only.

---

## 2. Prerequisites & Environment Setup

### System Dependencies

```bash
# pdftotext (from poppler-utils) — required by Agent 1
sudo apt-get install poppler-utils

# OpenCV system libs — required by Agent 3
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### Conda / Pip Environment

All agents must be run inside the `rag_prod` conda environment.

```bash
conda activate rag_prod
```

Install Python dependencies:

```bash
pip install watchdog requests chromadb rank-bm25 \
            pymupdf opencv-python numpy tqdm \
            layoutparser detectron2 \
            langchain-ollama langchain-experimental ollama \
            langgraph langgraph-prebuilt langchain-core \
            ragas datasets
```

> **Note:** `detectron2` installation can be version-sensitive. Follow the official [Detectron2 install guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) matching your CUDA version.

### Ollama Models

Pull the required models before running any agent:

```bash
ollama pull gemma4:latest
ollama pull nomic-embed-text
```

For RAG evaluation only:

```bash
ollama pull deepseek-r1:14b
```

Verify models are available:

```bash
ollama list
```

### Detectron2 Weights

Agent 3 uses a pre-trained layout detection model. The weight file must exist at:

```
../model_final.pth          # one level above the citation_agent/ directory
```

This is the `PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x` checkpoint. It is downloaded automatically by `layoutparser` on first use, or you can pre-place it manually.

---

## 3. Directory Structure

```
citation_agent/
├── raw/                        # ← DROP your source PDFs here (Agent 1 reads from here)
├── pulled_pdfs/                # Auto-created by Agent 2 — downloaded reference PDFs
├── drafts/                     # ← DROP your .txt draft files here (Orchestrator watches this)
├── physics_vectordb/           # ChromaDB persistent vector database (auto-created by Agent 3)
├── extracted_citations.json    # Output of Agent 1
├── downloaded.json             # Output of Agent 2 — filepath → citation string map
├── failed_downloads.json       # Output of Agent 2 — failed citation list with reasons
├── bm25_index.pkl              # BM25 sparse index rebuilt after every Agent 3 run
├── sample_inputs               # Sample queries for RAG evaluation
├── evaluation_results.csv      # Output of evaluate_rag.py
├── agent1_extractor.py
├── agent2_fetcher.py
├── agent3_ingestor.py
├── agent4_assistant.py
├── agent5_batch_citer.py
├── agent_graph.py              # LangGraph supervisor agent
├── master_orchestrator.py
└── evaluate_rag.py

../extracted_data/
└── images/                     # Cropped figure/table images extracted by Agent 3
```

> **Important:** The `physics_vectordb/`, `bm25_index.pkl`, `downloaded.json`, and `pulled_pdfs/` are excluded from git (see `.gitignore`). They are built locally and can be large.

---

## 4. Data Flow Diagram

```
raw/*.pdf
    │
    ▼  [Agent 1 — pdftotext + regex]
extracted_citations.json
    │
    ▼  [Agent 2 — Crossref → Unpaywall → arXiv fallback]
pulled_pdfs/*.pdf  +  downloaded.json  +  failed_downloads.json
    │
    ▼  [Agent 3 — Detectron2 layout + gemma4 VLM + nomic-embed-text]
physics_vectordb/  (ChromaDB)  +  bm25_index.pkl  +  ../extracted_data/images/
    │
    ├──▶  [Agent 4] Interactive per-sentence assistant  →  stdout suggestion
    │
    └──▶  [Agent 5] Batch file processor  →  <draft>_cited.txt  +  <draft>_citations.json

────────────────────────────────────────────────
Orchestration Layer (master_orchestrator.py):
  watchdog file monitors ──▶ agent_graph.py (LangGraph Supervisor)
                                 │
                                 ▼
                            gemma4:latest decides which tool to call
                                 │
                            ┌────┴────┐
                            ▼         ▼
                      Agents 1-3   Agent 5
                      (PDF flow)   (Draft flow)
```

---

## 5. Running the Full Pipeline (Automated — Recommended)

The `master_orchestrator.py` script runs as a persistent background daemon that watches two directories. When events occur, it delegates to the **LangGraph supervisor agent** (`agent_graph.py`), which uses `gemma4:latest` to reason about which tools (agents) to invoke.

### Start the Orchestrator

```bash
cd /path/to/citation_agent
conda activate rag_prod
python master_orchestrator.py
```

You will see:

```
Starting Master Orchestrator (Dual Mode)...
 - Monitoring 'raw/' for new PDFs (Cooldown: 30s, Workers: 4)
 - Monitoring 'drafts/' for text drafts (Cooldown: 2s)
Press Ctrl+C to stop.
```

### Ingesting New Papers (Agents 1–3)

1. Copy one or more source PDFs into the `raw/` directory.
2. The orchestrator detects the new file(s) immediately.
3. It starts a **30-second cooldown timer**, resetting it each time another PDF is dropped — this lets you batch-drop many files before the pipeline fires.
4. After the cooldown expires, it sends a natural-language message to the **LangGraph supervisor**, which autonomously decides to call `extract_citations_tool`, `fetch_papers_tool`, and `ingest_papers_tool` in sequence.

> **If a pipeline run is already in progress** when you drop more files, the new files are queued and processed in a second run automatically after the current one finishes.

### Auto-Citing Draft Files (Agent 5)

1. Copy or save a plain-text draft (`.txt`) into the `drafts/` directory.
2. The orchestrator detects it within 2 seconds.
3. It sends a message to the LangGraph supervisor, which calls `batch_cite_tool` with the file path.
4. Output is saved as `drafts/<your_file>_cited.txt` with a companion `drafts/<your_file>_citations.json` mapping.

### Stopping the Orchestrator

Press `Ctrl+C`. All pending timers are cancelled cleanly before exit.

---

## 6. Running Agents Manually (Step-by-Step)

All commands below must be run from inside the `citation_agent/` directory with `rag_prod` active.

```bash
cd /path/to/citation_agent
conda activate rag_prod
```

---

### Agent 1 — Extractor (`agent1_extractor.py`)

**What it does:**
- Scans every `*.pdf` in the `raw/` directory using `pdftotext`.
- Parses the References section by finding lines matching the pattern `[N] <text>`.
- Accumulates multi-line citations and deduplicates across all input PDFs.
- Saves the result to `extracted_citations.json`.

**Run:**

```bash
python agent1_extractor.py
```

**Input:** `raw/*.pdf` (one or more PDFs)

**Output:** `extracted_citations.json` — a JSON array of citation strings, e.g.:

```json
[
    "[1] A. Smith et al., \"Quantum Coherence\", Phys. Rev. Lett., 2021.",
    "[2] B. Jones, \"Dark Matter Survey\", arXiv:2103.01234, 2021."
]
```

**Notes:**
- Citations with fewer than 2 characters of page-number-like content are automatically skipped.
- All PDFs in `raw/` are processed together; duplicates across files are removed.
- There are no CLI arguments — source directory and output file are hardcoded.

---

### Agent 2 — Fetcher (`agent2_fetcher.py`)

**What it does:**
- Reads `extracted_citations.json` (skips entries > 500 chars, which are usually malformed).
- For each citation string, runs a **3-stage lookup**:
  1. **Crossref API** — finds a DOI and paper title via bibliographic search.
  2. **Unpaywall API** — checks if the DOI has an open-access PDF and downloads it.
  3. **arXiv fallback** — if Unpaywall fails, searches arXiv by title and downloads the PDF.
- Saves downloaded PDFs to `pulled_pdfs/`.
- Writes `downloaded.json` (filepath → citation mapping) and `failed_downloads.json`.

**Run:**

```bash
python agent2_fetcher.py
```

**Input:** `extracted_citations.json`

**Output:**
- `pulled_pdfs/<sanitised_title>_<index>.pdf` — Unpaywall downloads
- `pulled_pdfs/<sanitised_title>_<index>_arxiv.pdf` — arXiv fallback downloads
- `downloaded.json` — `{ "pulled_pdfs/filename.pdf": "citation string", ... }`
- `failed_downloads.json` — `[ { "citation": "...", "reason": "..." }, ... ]`

**Rate limiting (built-in):**
- 0.5s sleep between successful Unpaywall downloads.
- 3s sleep between arXiv requests (arXiv enforces 1 req/3s).
- 3s sleep when falling through to arXiv after a failed Unpaywall lookup.

**Notes:**
- The Unpaywall API email is hardcoded as `researcher123987@gmail.com`. To use your own email (recommended for higher rate limits), edit line 21 of `agent2_fetcher.py`:
  ```python
  email = "your.email@example.com"
  ```
- Papers behind a paywall with no arXiv preprint will appear in `failed_downloads.json` with the reason `"Paywalled / Not Open Access"`.

---

### Agent 3 — Ingestor (`agent3_ingestor.py`)

**What it does:**
- Reads `downloaded.json` to get the list of PDFs to process.
- For each PDF, uses **Detectron2** (`PubLayNet` model) to detect layout blocks (text, figures, tables).
- Extracts raw text blocks via PyMuPDF (`fitz`).
- For each figure/table detected, crops the image and calls **`gemma4:latest`** (multimodal) to generate a 3–5 sentence description.
- Applies **SemanticChunker** (LangChain, backed by `nomic-embed-text`) to split text into meaningful chunks.
- Deduplicates chunks and upserts them into a **ChromaDB** persistent collection (`physics_papers`).
- Rebuilds the **BM25 sparse index** from all chunks in the database and saves it to `bm25_index.pkl`.

**Run (sequential, default):**

```bash
python agent3_ingestor.py
```

**Run with parallel workers (recommended for large batches):**

```bash
python agent3_ingestor.py --workers 4
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--workers` | int | `1` | Number of parallel PDF processing workers. Use `1` for sequential (safer for GPU memory), `2–4` for multiprocessing. |

**Input:** `downloaded.json`, `pulled_pdfs/*.pdf`

**Output:**
- `physics_vectordb/` — ChromaDB persistent vector database (created/updated)
- `bm25_index.pkl` — BM25 sparse index rebuilt from the full database
- `../extracted_data/images/` — cropped figure/table PNG images

**Important notes:**
- Detectron2 weights are loaded at `../model_final.pth` (relative to `citation_agent/`).
- Each worker initialises its own Detectron2 model instance; use `--workers 1` if you run into GPU OOM errors.
- The ingestor **appends** to an existing ChromaDB collection — it does not wipe and recreate it. Running it multiple times on new PDFs is safe and incremental.
- Chunk IDs are sequentially numbered (`chunk_0`, `chunk_1`, …) and the ingestor reads the current max index before adding new chunks.
- Text chunks shorter than 10 characters are silently discarded.
- Embeddings are batched in groups of 1000, with each document truncated to 4000 characters max.

---

### Agent 4 — Interactive Assistant (`agent4_assistant.py`)

**What it does:**
- Loads the ChromaDB collection and the BM25 index.
- Takes a **single draft sentence** as input.
- Performs **Hybrid Search** (BM25 sparse + `nomic-embed-text` dense, fused via Reciprocal Rank Fusion at `k=60`).
- Sends the retrieved context + citations to **`gemma4:latest`** which rewrites the sentence with an inline citation and explains why.
- Prints token usage stats (prompt tokens submitted, tokens generated).

**Run:**

```bash
python agent4_assistant.py --text "Your draft sentence goes here."
```

| Flag | Type | Default | Required | Description |
|------|------|---------|----------|-------------|
| `--text` | str | — | **Yes** | The draft sentence or passage to find a citation for. |
| `--top_k` | int | `3` | No | Number of context chunks to retrieve and pass to the LLM. |

**Example:**

```bash
python agent4_assistant.py \
  --text "Dark matter accounts for approximately 27% of the universe's total energy density." \
  --top_k 5
```

**Sample output:**

```
Connecting to Vector DB and BM25...
Searching DB for relevant context for query: 'Dark matter accounts for...'
--- Found References ---
 > [3] Planck Collaboration, "Planck 2018 results...", A&A, 2020.

Drafting citation suggestion...

[Token Stats] Submitted: 1842 | Generated: 97

=== AI ASSISTANT SUGGESTION ===
Dark matter accounts for approximately 27% of the universe's total energy density [3].
The Planck 2018 results provide precise cosmological parameters including the dark matter density...
===============================
```

**Notes:**
- Agent 4 is **read-only** — it never modifies the database.
- The `hybrid_search` function defined in this file is also imported by Agent 5 and `evaluate_rag.py`.
- The hybrid search retrieves `max(15, top_k * 3)` candidates from each retrieval method before fusing, so increasing `--top_k` also broadens the initial candidate pool.
- The agent must be run from the `citation_agent/` directory so it can locate `./physics_vectordb` and `./bm25_index.pkl`.

---

### Agent 5 — Batch Citer (`agent5_batch_citer.py`)

**What it does:**
- Reads a plain `.txt` draft file.
- Splits it into individual sentences using a regex splitter (on `.`, `!`, `?`).
- For each sentence (longer than 3 words), asks **`gemma4:latest`** whether it is a factual scientific claim requiring a citation (`YES`/`NO`).
- For sentences that need citations, runs the same **Hybrid Search** as Agent 4 (`top_k=3`).
- Builds a running citation key map (`cite_1`, `cite_2`, …) — the same source always gets the same key within a single run.
- Asks `gemma4:latest` to rewrite the sentence appending `\cite{cite_key}`.
- Joins all sentences back into a cited draft and saves it.
- Saves a companion `_citations.json` mapping (`cite_key` → full citation string).
- Prints token stats per sentence.

**Run:**

```bash
python agent5_batch_citer.py --file drafts/my_draft.txt --out drafts/my_draft_cited.txt
```

| Flag | Type | Default | Required | Description |
|------|------|---------|----------|-------------|
| `--file` | str | — | **Yes** | Path to the input plain-text draft. |
| `--out` | str | `cited_draft.txt` | No | Path for the output cited draft. Defaults to current directory. |

**Output files:**

| File | Contents |
|------|----------|
| `drafts/introduction_cited.txt` | Full draft with `\cite{cite_key}` tags inserted inline |
| `drafts/introduction_citations.json` | `{ "cite_1": "[3] Planck Collaboration ...", "cite_2": "..." }` |

**Notes:**
- Sentences shorter than 4 words are **always skipped** (no citation check performed).
- If no relevant context is found in the database for a sentence, it is passed through unchanged.
- Agent 5 imports `hybrid_search` directly from `agent4_assistant.py`, so both scripts must be in the same directory.
- When run via the orchestrator, the output path is automatically set to `<input_path>_cited.txt` and the mapping to `<input_path>_citations.json`.

---

## 7. LangGraph Supervisor Agent

### Overview (`agent_graph.py`)

The orchestrator no longer calls agents directly via subprocess. Instead, it delegates to a **LangGraph ReAct-style agent** that uses `gemma4:latest` (with tool-calling) to autonomously decide which pipeline steps to run.

### How It Works

1. The orchestrator sends a natural-language message (e.g. *"New PDFs have been added to the raw/ directory…"*) to `process_event()`.
2. `process_event()` feeds this message into a compiled LangGraph `StateGraph`.
3. The graph has two nodes:
   - **`agent`** — calls `gemma4:latest` (with tools bound) to reason and decide the next action.
   - **`tools`** — executes the selected tool and returns the result.
4. The graph loops (`agent → tools → agent → …`) until the LLM decides no more tools are needed, at which point it emits a final text response and the graph terminates.

### Available Tools

| Tool Name | Wraps | Description |
|-----------|-------|-------------|
| `extract_citations_tool` | `agent1_extractor.run_extractor()` | Extracts citations from PDFs in `raw/` |
| `fetch_papers_tool` | `agent2_fetcher.fetch_papers()` | Downloads referenced papers |
| `ingest_papers_tool` | `agent3_ingestor.run_ingestor(workers)` | Ingests papers into ChromaDB + BM25 (default 4 workers) |
| `batch_cite_tool` | `agent5_batch_citer.run_batch_citer(path)` | Cites a draft text file |

### Running Standalone

You can invoke the agent graph directly without the file watchers:

```bash
python agent_graph.py
```

This sends a default message (*"A new PDF was dropped in the raw directory. Please process it."*) to the supervisor. You can also import and call `process_event()` from your own scripts:

```python
from agent_graph import process_event
process_event("Please extract citations and fetch the papers.")
```

---

## 8. RAG Evaluation (Ragas)

### Overview (`evaluate_rag.py`)

An automated evaluation script that measures the quality of the RAG pipeline using the [Ragas](https://docs.ragas.io/) framework.

### What It Does

1. Loads the ChromaDB collection and BM25 index.
2. Parses sample queries from the `sample_inputs` file (queries separated by blank lines).
3. For each query, runs the full RAG pipeline (hybrid search → `gemma4:latest` generation).
4. Evaluates the responses using **`deepseek-r1:14b`** as the judge LLM with two metrics:
   - **Faithfulness** — is the answer grounded in the retrieved context?
   - **Answer Relevancy** — is the answer relevant to the question?
5. Saves detailed results to `evaluation_results.csv`.

### Prerequisites

```bash
ollama pull deepseek-r1:14b
```

### Run

```bash
python evaluate_rag.py
```

**Input:** `sample_inputs` (plain text, queries separated by blank lines)

**Output:** `evaluation_results.csv` — per-query faithfulness and answer relevancy scores.

### Configuration

| Variable | Line | Description |
|----------|------|-------------|
| Evaluator LLM | 130 | `deepseek-r1:14b` — change to use a different judge model |
| Evaluator embeddings | 131 | `nomic-embed-text:latest` |
| `timeout` | 133 | 1800s (30 min) — increase for very large evaluation sets |
| `max_retries` | 133 | 5 retries per evaluation call |

---

## 9. Output Files Reference

| File | Created by | Description |
|------|-----------|-------------|
| `extracted_citations.json` | Agent 1 | JSON array of raw reference strings from source PDFs |
| `downloaded.json` | Agent 2 | `{ "pdf_path": "citation_string" }` for successful downloads |
| `failed_downloads.json` | Agent 2 | Array of `{ "citation", "reason" }` for failed downloads |
| `physics_vectordb/` | Agent 3 | ChromaDB persistent store — contains all text chunks, figure descriptions, and their embeddings |
| `bm25_index.pkl` | Agent 3 | Pickled `BM25Okapi` object — rebuilt from the full database after every Agent 3 run |
| `../extracted_data/images/` | Agent 3 | Cropped figure/table PNG images named `<pdf>_p<page>_f<fig_idx>.png` |
| `<draft>_cited.txt` | Agent 5 | Draft with `\cite{key}` tags inserted |
| `<draft>_citations.json` | Agent 5 | Maps `cite_N` keys → full citation strings for use in a BibTeX builder |
| `evaluation_results.csv` | `evaluate_rag.py` | Per-query Ragas evaluation scores |

---

## 10. Configuration Reference

Hardcoded constants that can be changed directly in the scripts:

### `master_orchestrator.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `COOLDOWN_SECONDS` | `30` | Seconds to wait after the last PDF drop before firing the pipeline |
| `DRAFT_COOLDOWN_SECONDS` | `2` | Debounce delay before running Agent 5 on a modified draft |
| `WORKERS` | `4` | Number of parallel workers passed to Agent 3 |
| `RAW_DIR` | `"raw"` | Directory watched for new source PDFs |
| `DRAFTS_DIR` | `"drafts"` | Directory watched for new/modified draft `.txt` files |

### `agent_graph.py`

| Variable | Default | Description |
|----------|---------|-------------|
| LLM model | `gemma4:latest` | The model used by the supervisor agent for tool-calling decisions |
| `temperature` | `0` | Deterministic tool selection |
| `ingest_papers_tool` workers | `4` | Default worker count when the supervisor calls the ingest tool |

### `agent2_fetcher.py`

| Variable | Line | Description |
|----------|------|-------------|
| `email` | 21 | Email for the Unpaywall API — replace with your own |

### `agent3_ingestor.py`

| Variable | Line | Description |
|----------|------|-------------|
| `detectron_weights` | 159 | Path to `model_final.pth` — defaults to `../model_final.pth` |
| `images_dir` | 160 | Where cropped figures are saved — defaults to `../extracted_data/images` |
| `breakpoint_threshold_amount` | 189 | SemanticChunker threshold (90th percentile) — lower = more chunks |
| `batch_size` | 266 | ChromaDB embedding batch size — reduce if hitting memory limits |

### `agent4_assistant.py`

| Variable | Line | Description |
|----------|------|-------------|
| `rrf_k` | 14 | RRF fusion constant (default 60) — higher = less rank-bias |

---

## 11. Troubleshooting

### `pdftotext: command not found`
```bash
sudo apt-get install poppler-utils
```

### Agent 3 crashes with CUDA OOM
- Reduce `--workers` to `1` or `2`.
- The Detectron2 model is loaded per worker, so each worker consumes GPU VRAM independently.

### `No module named 'watchdog'`
```bash
pip install watchdog
```

### `No module named 'langgraph'`
```bash
pip install langgraph langgraph-prebuilt
```

### ChromaDB collection `physics_papers` not found (Agent 4 / 5)
Agent 3 must be run at least once to create the database before Agents 4 or 5 can query it.

### BM25 index shape mismatch
The BM25 index is rebuilt by Agent 3 from the full database. If you manually edit ChromaDB, re-run Agent 3 with `--workers 1` on an empty `downloaded.json` to force a BM25 rebuild without ingesting new data.

### arXiv rate limiting (`HTTP 429`)
Agent 2 enforces a 3-second sleep between arXiv requests. If you still hit limits (e.g. running multiple instances), increase the `time.sleep(3)` values on lines 114 and 134 of `agent2_fetcher.py`.

### `ollama: connection refused`
Make sure the Ollama daemon is running:
```bash
ollama serve
```

### Agent 5 — All sentences marked "NO citation needed"
This typically means `gemma4:latest` is being cautious. You can lower the threshold by modifying the prompt in the `needs_citation()` function (line 16 of `agent5_batch_citer.py`) to be more permissive, or run Agent 4 interactively on specific sentences instead.

### LangGraph supervisor loops or picks wrong tools
The supervisor relies on `gemma4:latest` tool-calling. If it loops or selects incorrect tools, try:
- Ensuring Ollama is serving the latest `gemma4` weights (`ollama pull gemma4:latest`).
- Running the agents manually (Section 6) to bypass the supervisor entirely.
