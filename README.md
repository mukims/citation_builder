# Citation Agent RAG Pipeline

A comprehensive, fully automated Retrieval-Augmented Generation (RAG) pipeline designed to automate the extraction, fetching, ingestion, and citation-assistance of scientific literature using local models. 

This system extracts reference strings from a base PDF, automatically downloads open-access versions of the referenced papers, ingests both text and figures into a local Vector Database, and automatically injects proper LaTeX citations into your text drafts.

## System Architecture

The pipeline consists of five sequential agents, managed by an intelligent background orchestrator and a LangGraph-based supervisor agent.

### Orchestration & Control Flow
- **Master Orchestrator (`master_orchestrator.py`)**: Continuously monitors the `raw/` directory for new PDFs and the `drafts/` directory for text drafts. It implements debouncing and cooldown timers to prevent GPU memory crashes during rapid file drops, and delegates events to the LangGraph supervisor.
- **LangGraph Supervisor (`agent_graph.py`)**: A ReAct-style agent utilizing LangGraph and `gemma4:latest` with tool-calling capabilities. When an event is triggered by the orchestrator (e.g., "A new PDF was dropped"), this agent autonomously reasons about the state of the pipeline and iteratively calls the necessary underlying agent tools to process the data.

### Data Ingestion (Agents 1-3)
1. **Agent 1: Extractor (`agent1_extractor.py`)**: 
   - Scans all PDFs dropped into the `raw/` directory using `pdftotext`.
   - Uses regex parsing to locate the "References" section and extract individual citation strings (e.g., `[1] Author, Title...`).
   - Deduplicates citations and outputs them to `extracted_citations.json`.
   
2. **Agent 2: Fetcher (`agent2_fetcher.py`)**: 
   - Reads the extracted citations and performs a 3-stage lookup to find open-access full texts.
   - **Stage 1**: Queries the Crossref API to resolve the citation string to a DOI and metadata.
   - **Stage 2**: Queries the Unpaywall API to check for an open-access PDF link.
   - **Stage 3**: If Unpaywall fails, falls back to searching the arXiv API by title.
   - Downloads successful matches to `pulled_pdfs/` and maintains state in `downloaded.json` and `failed_downloads.json`.

3. **Agent 3: Ingestor (`agent3_ingestor.py`)**: 
   - A multimodal processing engine that reads the downloaded PDFs.
   - Uses **Detectron2** (`PubLayNet`) to detect document layouts (text blocks, figures, tables).
   - Crops figures and tables, and passes them to **`gemma4:latest`** (multimodal) to generate rich textual descriptions of the visual data.
   - Splits text using SemanticChunker, embeds everything using **`nomic-embed-text`**, and upserts it into a **ChromaDB** persistent vector database.
   - Automatically rebuilds a sparse **BM25 index** across the entire database for hybrid search.

### Inference & Writing (Agents 4-5)
4. **Agent 4: Assistant (`agent4_assistant.py`)**: 
   - An interactive CLI assistant for citing individual sentences.
   - Performs a **Hybrid Search** (ChromaDB dense vectors + BM25 sparse index), fused via Reciprocal Rank Fusion (RRF) at `k=60`.
   - Passes the retrieved context and the user's sentence to `gemma4:latest`, which suggests a rewrite with an inline citation and explains its reasoning.

5. **Agent 5: Batch Citer (`agent5_batch_citer.py`)**: 
   - An automated draft processor that reads a plain `.txt` draft and splits it into sentences.
   - Semantically analyzes each sentence with `gemma4:latest` to determine if it contains a factual scientific claim that *requires* a citation.
   - For sentences needing citations, it performs the Hybrid Search, builds a continuous `cite_key` mapping, and rewrites the sentence to append LaTeX `\cite{cite_key}` tags.
   - Outputs a fully cited `_cited.txt` draft and a `_citations.json` mapping for BibTeX compilation.

### Evaluation
- **RAG Evaluation (`evaluate_rag.py`)**: Programmatic evaluation of the pipeline's retrieval and generation capabilities. It uses the **Ragas** framework alongside `deepseek-r1:14b` as a judge LLM to evaluate sample queries on metrics such as *Faithfulness* (is the answer grounded in context?) and *Answer Relevancy* (does it address the prompt?).

## 🚀 Usage

For full instructions on how to configure your environment and run the orchestrator, please refer to the **[Main USAGE Guide](USAGE.md)**.
