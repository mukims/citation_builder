# Latest Changes in Citation Agent

This document summarizes the recent updates and enhancements made to the `citation_agent` repository.

## 0. Architecture Refactoring & Reliability Improvements (Latest)
**Date:** May 17, 2026
**Summary:** Major refactoring to eliminate code duplication, fix critical bugs, and improve reliability across the entire pipeline.

### Bug Fixes (Critical)
- **Agent 2 `downloaded.json` overwrite bug fixed** — previously, every Agent 2 run would overwrite the file, silently destroying fetch history from prior runs. Now merges with existing state on startup.
- **Agent 2 crash-safe checkpointing** — `downloaded.json` and `failed_downloads.json` are now written to disk after every single paper, so crashes never lose progress.
- **Agent 2 skip-already-processed** — citations already in `downloaded.json` or `failed_downloads.json` are skipped on re-runs, making the fetcher fully incremental.

### New: `config.py` — Central Configuration
- All hardcoded model names (`gemma4:latest`, `nomic-embed-text`, etc.), file paths, directory paths, and tunable constants moved to a single `config.py`.
- Changing a model or threshold now requires editing one file instead of 5+.

### New: `shared/` Module
- **`shared/ingestion.py`** — Extracted ~200 lines of duplicated PDF processing (Detectron2 + VLM + ChromaDB + BM25) shared by Agent 3 and Agent 6.
- **`shared/search.py`** — Hybrid search with a singleton `OllamaEmbeddings` instance. Previously, Agent 5 created a new connection per sentence (~50 connections for a 50-sentence draft).
- **`shared/db.py`** — ChromaDB + BM25 loading boilerplate extracted from Agents 4, 5, and `evaluate_rag.py`.
- **`shared/retry.py`** — Exponential-backoff retry decorator wrapping all Ollama calls (3 attempts, 2s/4s/8s backoff).
- **`shared/log.py`** — Python `logging` module replacing all `print()` calls. Console + rotating log file (`logs/citation_agent.log`, 5 MB, 3 backups).

### Agent Improvements
- **Agent 1** — Now supports multiple reference formats (`[N] Author...` and `N. Author...`) via configurable regex patterns.
- **Agent 5** — Improved sentence splitter handles `et al.`, `Fig.`, `Eq.`, `Dr.`, `i.e.` without breaking.
- **Agent 5** — Batched citation-need check: all sentences checked in a single LLM call instead of one per sentence (cuts LLM round-trips in half).
- **All agents** — Bare `except:` clauses replaced with specific `except (ValueError, IndexError)`.
- **All agents** — LLM calls wrapped with retry decorator.

### Code Reduction
- Agent 3: 315 → 65 lines (delegates to `shared/ingestion.py`)
- Agent 6: 397 → 150 lines (delegates to `shared/ingestion.py`)
- Agent 4: 162 → 80 lines (delegates to `shared/search.py` + `shared/db.py`)

### Documentation
- `README.md`, `USAGE.md`, `FAQ.md` updated to reflect all changes.

---

## 1. Architectural Refactoring & Agent Graph (Latest)
**Commit:** `d6e82f1`
**Date:** May 8, 2026
**Summary:** The core architecture of the citation agent has been significantly refactored. The pipeline has been shifted towards a proper agentic framework.
- **Agent Graph Introduced:** A new file, `agent_graph.py`, was introduced to represent the agentic control flow and orchestration logic.
- **CLI Modularization:** `agent3_ingestor.py` and `agent5_batch_citer.py` have been updated with modularized CLI interfaces, enabling cleaner parameter passing and execution.
- **Failure Tracking Updates:** `failed_downloads.json` has seen substantial updates reflecting improved logging and tracking of problematic document downloads.
- **Orchestrator Enhancements:** The `master_orchestrator.py` was tweaked to integrate with the new architectural patterns.

## 2. RAG Evaluation Pipeline Setup
**Commit:** `c9332db`
**Date:** May 8, 2026
**Summary:** An evaluation framework for the RAG (Retrieval-Augmented Generation) components was integrated using the Ragas framework.
- **New Evaluation Script:** Created `evaluate_rag.py` to programmatically measure metrics like faithfulness, answer relevance, and context precision.
- **Evaluation Results:** Added `evaluation_results.csv` to persistently store the results of the automated evaluations.
- **Sample Inputs Support:** Added `sample_inputs` to help standardize the evaluation and testing of the citation agent.
- **Assistant Adjustments:** Minor modifications were made to `agent4_assistant.py` to support testing and evaluation loops.

## 3. Comprehensive Documentation Update
**Commit:** `e4dd749`
**Date:** April 27, 2026
**Summary:** The usage documentation for the repository was vastly expanded.
- **`USAGE.md` Rewritten:** A simple placeholder was replaced with a highly detailed, comprehensive usage guide (over 500 lines added) that provides step-by-step instructions for running the multi-agent pipeline and understanding the overarching workflow.

---
*Note: There are also some unstaged changes currently present in `extracted_citations.json`.*
