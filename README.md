# Citation Agent Multi-modal RAG Pipeline

A comprehensive, fully automated Retrieval-Augmented Generation (RAG) pipeline designed to automate the extraction, fetching, ingestion, and citation-assistance of scientific literature using local models. 

This system extracts reference strings from a base PDF, automatically downloads open-access versions of the referenced papers, ingests both text and figures into a local Vector Database, and automatically injects proper LaTeX citations into your text drafts.

## System Architecture

The pipeline consists of seven agents, a shared utility layer, and an orchestrator that watches the working directories and runs the appropriate stages.

### Core Infrastructure

- **Central Configuration (`config.py`)**: A single source of truth for all model names, file paths, directory locations, and tunable constants. Changing a model or threshold only requires editing one file.
- **Prompts (`prompts.py`)**: Every prompt sent to a model, in one place. Notably, Agent 4 and `evaluate_rag.py` share one system prompt rather than holding copies — the evaluation is only meaningful while it scores the prompt the agent actually uses.
- **Shared Utilities (`shared/`)**: A reusable library of modules shared across all agents:
  - `shared/ingestion.py` — `ingest_pdfs()`, the single ingestion path (process → upsert → mark → index) used by Agents 3 and 6 and the orchestrator, plus the Detectron2/VLM processing and BM25 rebuild underneath it
  - `shared/search.py` — Hybrid search (dense + sparse) with Reciprocal Rank Fusion and singleton embeddings
  - `shared/db.py` — ChromaDB + BM25 loading utilities
  - `shared/retry.py` — Exponential-backoff retry decorator for Ollama calls
  - `shared/log.py` — Centralised `logging` (console, plus a lazily-opened rotating file; see `CITATION_LOG_DIR` / `CITATION_LOG_FILE`)

### Orchestration & Control Flow
- **Master Orchestrator (`master_orchestrator.py`)**: The single entrypoint. Continuously monitors the `raw/`, `drafts/`, and `pulled_pdfs/` directories, with debouncing and cooldown timers to prevent GPU memory crashes during rapid file drops, and runs the pipeline stages directly:
  - `raw/` → Agent 1 (extract) → Agent 2 (fetch) → Agent 3 (ingest)
  - `pulled_pdfs/` → batched ingest via `shared/ingestion.py`
  - `drafts/` → Agent 5 (batch cite)

  These sequences are fixed, so they are called directly rather than planned by an LLM. An earlier LangGraph supervisor (`agent_graph.py`) chose the stages at runtime; it now lives in `attic/` and nothing imports it.

### Data Ingestion (Agents 1-3)
1. **Agent 1: Extractor (`agent1_extractor.py`)**: 
   - Scans all PDFs dropped into the `raw/` directory using `pdftotext`.
   - Supports multiple reference formats (`[N] Author...` and `N. Author...`) via configurable regex patterns.
   - Deduplicates citations and outputs them to `extracted_citations.json`.
   - Files each source PDF by outcome: `raw/processed/` when references were read, `raw/failed/` when nothing could be extracted (a scan with no text layer, or an unrecognised reference format), so a paper is never silently swallowed.
   
2. **Agent 2: Fetcher (`agent2_fetcher.py`)**: 
   - Reads the extracted citations and performs a 3-stage lookup to find open-access full texts.
   - **Stage 1**: Queries the Crossref API to resolve the citation string to a DOI and metadata.
   - **Stage 2**: Queries the Unpaywall API to check for an open-access PDF link.
   - **Stage 3**: If Unpaywall fails, falls back to searching the arXiv API by title.
   - Downloads successful matches to `pulled_pdfs/` and maintains state in `downloaded.json` and `failed_downloads.json`.
   - **Names files by paper identity** (DOI → arXiv id → title), so two citation strings that resolve to the same work reuse one download instead of fetching and re-parsing it twice.
   - **Incremental & crash-safe**: Merges with existing state on startup and checkpoints after every paper.

3. **Agent 3: Ingestor (`agent3_ingestor.py`)**: 
   - A multimodal processing engine that reads the downloaded PDFs.
   - Delegates to `shared/ingestion.py` for layout detection (Detectron2 PubLayNet), VLM figure description (`gemma4:latest`), semantic chunking, and ChromaDB upsert.
   - Rebuilds the sparse **BM25 index** once per batch, and only when the collection actually changed — a full rebuild re-tokenises every chunk in the database.

### Inference & Writing (Agents 4-5)
4. **Agent 4: Assistant (`agent4_assistant.py`)**: 
   - An interactive CLI assistant for citing individual sentences.
   - Uses `shared/search.py` for **Hybrid Search** (dense + BM25 sparse, fused via RRF).
   - LLM calls are wrapped with automatic retry logic.

5. **Agent 5: Batch Citer (`agent5_batch_citer.py`)**: 
   - An automated draft processor that reads a plain `.txt` draft.
   - Uses an improved sentence splitter that handles scientific abbreviations (`et al.`, `Fig.`, `Eq.`).
   - **Batched citation-need check**: Determines which sentences need citations in a single LLM call instead of one per sentence.
   - Outputs a fully cited `_cited.txt` draft and a `_citations.json` mapping for BibTeX compilation.

### Manual Ingestion
6. **Agent 6: Manual Ingestor (`agent6_manual_ingestor.py`)**:
   - Watches `pulled_pdfs/` for manually dropped PDFs and ingests them directly.
   - Delegates to `shared.ingestion.ingest_pdfs()`, the same path Agent 3 uses, so a re-dropped paper is skipped rather than re-parsed.

### Research Chat
7. **Agent 7: Research Chat (`agent7_research_chat.py`)**:
   - A multi-turn conversational RAG agent over the ingested corpus, for brainstorming and literature questions rather than citation insertion.
   - Retrieves via `shared/search.py` (default `top_k=5`) and streams responses from `CHAT_MODEL` (`qwen2.5:7b`) using the tuned `CHAT_OLLAMA_OPTIONS`.
   - Supports `/clear`, `/sources`, `/export`, and `/help` session commands.

### Evaluation
- **RAG Evaluation (`evaluate_rag.py`)**: Programmatic evaluation of the pipeline's retrieval and generation capabilities. Uses the **Ragas** framework alongside `deepseek-r1:14b` as a judge LLM to evaluate sample queries on metrics such as *Faithfulness* and *Answer Relevancy*.

## Getting Started

```bash
conda env create -f environment.yml && conda activate rag_prod
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
ollama pull gemma4:latest && ollama pull nomic-embed-text
python master_orchestrator.py
```

Then drop a source PDF into `raw/` and a `.txt` draft into `drafts/`.

Dependency versions are pinned in `environment.yml`; `requirements.txt` is the
pip equivalent, with `requirements-eval.txt` for the Ragas evaluation.
Detectron2 is installed separately because it is not on PyPI and is built
against your specific torch/CUDA.

For full instructions, see the **[Main USAGE Guide](USAGE.md)**; the
**[FAQ](FAQ.md)** covers architecture and troubleshooting.

## Development

```bash
python -m pip install -r requirements-test.txt
CITATION_LOG_FILE=0 python -m unittest discover -s . -p "test_*.py"
```

The test suite needs only three light packages — ChromaDB, PyTorch, detectron2
and the LangChain embedding chain are imported lazily inside the functions the
tests never call, so the suite runs anywhere. CI runs it on Python 3.10 and
3.12 on every push.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `UNPAYWALL_EMAIL` | *(placeholder)* | Contact address Unpaywall requires on every request. Set this or stage 2 fails and every lookup falls back to arXiv. |
| `CITATION_IMAGES_DIR` | `images/` | Where figure crops are written during ingestion. They are handed to the VLM and never read back, so the directory is a debugging artefact and safe to delete. |
| `CITATION_LOG_DIR` | `logs/` | Where the rotating log file is written. |
| `CITATION_LOG_FILE` | `1` | Set to `0` for console-only logging. |

### `attic/`

Components kept for reference but wired into nothing — the retired LangGraph
supervisor among them. See [`attic/README.md`](attic/README.md).
