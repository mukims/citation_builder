# Latest Changes in Citation Agent

This document summarizes the recent updates and enhancements made to the `citation_agent` repository.

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
