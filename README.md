# Citation Agent RAG Pipeline

*This directory contains the Agentic Pipeline of the repository. For the overarching repository documentation, please see the [root README](../README.md).*

A comprehensive, fully automated Retrieval-Augmented Generation (RAG) pipeline designed to automate the extraction, fetching, ingestion, and citation-assistance of scientific literature using local models. 

This system extracts reference strings from a base PDF, automatically downloads open-access versions of the referenced papers, ingests both text and figures into a local Vector Database, and automatically injects proper LaTeX citations into your text drafts.

## System Architecture

The pipeline consists of five sequential agents, managed by an intelligent background orchestrator.

### The Master Orchestrator (`master_orchestrator.py`)
- Continuously monitors the `raw/` directory for new PDFs and the `drafts/` directory for text drafts.
- Automatically handles batching and cooldowns to prevent GPU memory crashes.
- Drives the entire pipeline (Agents 1-3, and Agent 5) completely hands-off.

### Data Ingestion (Agents 1-3)
1. **Agent 1: Extractor (`agent1_extractor.py`)**: Parses a base PDF to extract the Reference section.
2. **Agent 2: Fetcher (`agent2_fetcher.py`)**: Queries Crossref/Unpaywall APIs and falls back to arXiv to download open-access PDF links locally.
3. **Agent 3: Ingestor (`agent3_ingestor.py`)**: Parallelized multimodal processing of downloaded PDFs using `gemma4:latest` to describe figures, embedding everything into ChromaDB with `nomic-embed-text`.

### Inference & Writing (Agents 4-5)
4. **Agent 4: Assistant (`agent4_assistant.py`)**: Interactive CLI assistant. Performs hybrid search to provide a citation suggestion based on a drafted sentence.
5. **Agent 5: Batch Citer (`agent5_batch_citer.py`)**: Automated draft processor. Semantically processes sentences in a `.txt` draft, querying `gemma4` to decide if factual claims require citations, and automatically appending LaTeX `\cite{cite_key}` tags.

## 🚀 Usage

For full instructions on how to configure your environment and run the orchestrator, please refer to the **[Main USAGE Guide](../USAGE.md)** located in the root directory.
