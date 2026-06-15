# Citation Agent — Frequently Asked Questions (FAQ)

---

## General

### What is the Citation Agent?

The Citation Agent is a fully automated, multi-agent Retrieval-Augmented Generation (RAG) pipeline that takes raw scientific PDFs as input and produces properly cited LaTeX drafts as output. It extracts references, downloads open-access versions of cited papers, ingests their text and figures into a local vector database, and then uses that knowledge base to insert `\cite{}` tags into your writing.

### Who is this tool for?

Researchers, graduate students, and academic writers who want to automate the tedious process of finding and inserting citations into their manuscripts. It is especially useful for physics and adjacent STEM fields where references follow the `[N] Author, Title, Journal, Year` format.

### Does this require any cloud APIs or paid services?

No. The entire pipeline runs **locally** using [Ollama](https://ollama.com/) for LLM inference and [ChromaDB](https://www.trychroma.com/) for vector storage. The only external network calls are to free, open APIs (Crossref, Unpaywall, arXiv) during the paper-fetching stage. All model names and paths are centralised in a single `config.py` file.

### What LLMs does the system use?

| Model | Provider | Purpose |
|-------|----------|---------|
| `gemma4:latest` | Ollama | Vision-language figure description, citation judgement, response generation, and LangGraph supervisor reasoning |
| `nomic-embed-text` | Ollama | Dense text embeddings for ChromaDB and hybrid search |
| `deepseek-r1:14b` | Ollama | Judge LLM used **only** by the Ragas evaluation script |

All model names are configured in `config.py` and can be changed in one place.

---

## Architecture & Agents

### How many agents are there and what does each one do?

| Agent | Script | One-line summary |
|-------|--------|------------------|
| 1 — Extractor | `agent1_extractor.py` | Parses reference strings from source PDFs using `pdftotext` + regex |
| 2 — Fetcher | `agent2_fetcher.py` | Downloads open-access PDFs via Crossref → Unpaywall → arXiv fallback |
| 3 — Ingestor | `agent3_ingestor.py` | Multimodal ingestion (Detectron2 layout + VLM figure captions) into ChromaDB + BM25 |
| 4 — Assistant | `agent4_assistant.py` | Interactive CLI for single-sentence citation suggestions |
| 5 — Batch Citer | `agent5_batch_citer.py` | Automated full-draft citation processor that outputs `\cite{key}` tagged text |
| 6 — Manual Ingestor | `agent6_manual_ingestor.py` | Watches `pulled_pdfs/` for manually dropped PDFs and ingests them directly |

### What is the LangGraph Supervisor?

The supervisor (`agent_graph.py`) is a **ReAct-style agent** built with [LangGraph](https://langchain-ai.github.io/langgraph/) that wraps all agents as callable tools. Instead of running agents via subprocess, the orchestrator sends a natural-language event description (e.g., *"New PDFs were dropped in raw/"*) to the supervisor, which uses `gemma4:latest` with tool-calling to autonomously decide **which** agents to invoke and in **what order**.

### What is the Master Orchestrator?

`master_orchestrator.py` is a persistent background daemon that uses Python's `watchdog` library to monitor three directories:

| Directory | Trigger | Action |
|-----------|---------|--------|
| `raw/` | New PDF dropped | Debounce 30 s → delegate to LangGraph supervisor (Agents 1→2→3) |
| `drafts/` | New/modified `.txt` file | Debounce 2 s → delegate to LangGraph supervisor (Agent 5) |
| `pulled_pdfs/` | New PDF dropped | Debounce 5 s → Agent 6 ingests directly into ChromaDB |

### What is "hybrid search"?

Hybrid search combines two retrieval methods and fuses their results:

1. **Dense retrieval** — ChromaDB similarity search using `nomic-embed-text` embeddings.
2. **Sparse retrieval** — BM25 keyword matching over the full corpus.

Results from both are merged using **Reciprocal Rank Fusion (RRF)** at `k=60`, which balances precision from dense search with recall from sparse search. The implementation lives in `shared/search.py` and uses a **singleton embeddings model** — it is created once and reused across all calls, eliminating redundant connections.

### What is the `shared/` module?

The `shared/` directory contains reusable utilities that all agents import:

| Module | Purpose |
|--------|---------|
| `shared/ingestion.py` | PDF processing (Detectron2 layout + VLM), ChromaDB upsert, BM25 rebuild |
| `shared/search.py` | Hybrid search with singleton embeddings and RRF fusion |
| `shared/db.py` | ChromaDB + BM25 loading (replaces duplicated boilerplate in 3 agents) |
| `shared/retry.py` | Exponential-backoff retry decorator for Ollama calls |
| `shared/log.py` | Centralised logging to console + rotating log file |

### What is `config.py`?

A single file containing every model name, file path, directory path, and tunable constant in the pipeline. Instead of editing 5+ files to change a model, you edit one line in `config.py`.

### Why is Agent 6 separate from Agent 3?

Agent 3 ingests papers that were *automatically downloaded* by Agent 2 (tracked in `downloaded.json`). Agent 6 exists for papers that Agent 2 **couldn't** fetch — for example, paywalled papers you downloaded manually. Agent 6 watches `pulled_pdfs/` and ingests any PDF dropped there without requiring a corresponding entry in `downloaded.json`.

---

## Setup & Installation

### What are the system requirements?

- **OS:** Linux (tested on Ubuntu/Debian)
- **GPU:** NVIDIA GPU with CUDA support recommended (for Detectron2 layout detection and Ollama inference)
- **RAM:** 16 GB minimum; 32 GB recommended for parallel ingestion workers
- **Disk:** ~5 GB for models, plus storage for downloaded PDFs and ChromaDB

### How do I set up the environment?

```bash
# 1. Install system dependencies
sudo apt-get install poppler-utils libgl1-mesa-glx libglib2.0-0

# 2. Create and activate the conda environment
conda activate rag_prod

# 3. Install Python packages
pip install watchdog requests chromadb rank-bm25 \
            pymupdf opencv-python numpy tqdm \
            layoutparser detectron2 \
            langchain-ollama langchain-experimental ollama \
            langgraph langgraph-prebuilt langchain-core \
            ragas datasets

# 4. Pull required Ollama models
ollama pull gemma4:latest
ollama pull nomic-embed-text
```

### Where do I get the Detectron2 weights?

Agent 3 expects the PubLayNet `mask_rcnn_X_101_32x8d_FPN_3x` checkpoint at `../model_final.pth` (one directory above `citation_builder/`). It is downloaded automatically by `layoutparser` on first use, or you can pre-download and place it manually.

### Do I need a GPU?

A GPU is **strongly recommended** for:
- Detectron2 layout detection (Agent 3 / Agent 6)
- Ollama LLM inference (`gemma4`, `nomic-embed-text`)

The pipeline *can* run on CPU via Ollama's CPU mode, but ingestion and generation will be significantly slower.

---

## Usage

### What is the simplest way to use the pipeline?

1. Start the orchestrator: `python master_orchestrator.py`
2. Drop your source PDF(s) into the `raw/` directory.
3. Wait for the pipeline to finish (Agents 1→2→3 run automatically).
4. Drop your draft `.txt` file into the `drafts/` directory.
5. Pick up your cited draft at `drafts/<your_file>_cited.txt`.

### Can I run agents individually without the orchestrator?

Yes. Every agent is a standalone Python script:

```bash
python agent1_extractor.py                           # Extract citations from raw/*.pdf
python agent2_fetcher.py                             # Download referenced papers
python agent3_ingestor.py --workers 4                # Ingest into ChromaDB
python agent4_assistant.py --text "Your sentence."   # Interactive single-sentence mode
python agent5_batch_citer.py --file drafts/draft.txt # Batch-cite a full draft
python agent6_manual_ingestor.py --once paper.pdf    # Ingest a single manual PDF
```

### How do I add a paper that Agent 2 couldn't download?

Simply drop the PDF into the `pulled_pdfs/` directory. If the orchestrator is running, Agent 6 will automatically detect and ingest it within 5 seconds. Alternatively, run Agent 6 directly:

```bash
python agent6_manual_ingestor.py --once pulled_pdfs/my_paper.pdf --citation "Smith et al., Nature, 2024"
```

### What format should my draft be in?

A plain `.txt` file with standard English sentences. Agent 5 splits on sentence-ending punctuation (`. ! ?`) and processes each sentence independently. No special markup is required.

### What does the output look like?

Agent 5 produces two files:

**`draft_cited.txt`** — your text with `\cite{cite_N}` tags:
```
Dark matter accounts for approximately 27% of the total energy density \cite{cite_1}.
```

**`draft_citations.json`** — a key-to-reference mapping:
```json
{
  "cite_1": "[3] Planck Collaboration, \"Planck 2018 results...\", A&A, 2020."
}
```

You can use the JSON mapping to build your BibTeX bibliography.

### Does Agent 5 handle abbreviations like "et al." and "Fig."?

Yes. The sentence splitter uses a case-insensitive period masking and restoration technique for common scientific abbreviations (such as `et al.`, `Fig.`, `Eq.`, `Dr.`, `i.e.`, `e.g.`, `Ref.`, etc.). This avoids incorrect sentence breaks when these abbreviations are used inside sentences.

#### Example Test Cases & Expected Splits

Here is how the sentence splitter parses different inputs (verified by our automated test suite `test_sentence_splitter.py`):

1. **Standard Scientific Abbreviations & Casing:**
   - **Input:** `"See Fig. 1 for details. This is the next sentence. Also check fig. 2."`
   - **Expected Splits:**
     - `"See Fig. 1 for details."`
     - `"This is the next sentence."`
     - `"Also check fig. 2."`

2. **Equations and Plurals:**
   - **Input:** `"Using Eq. 3, we obtain the limit. Compare with eq. 4 and eqs. 5-6."`
   - **Expected Splits:**
     - `"Using Eq. 3, we obtain the limit."`
     - `"Compare with eq. 4 and eqs. 5-6."`

3. **Multi-Period Abbreviations:**
   - **Input:** `"This is e.g. a classic example. That is i.e. the only explanation."`
   - **Expected Splits:**
     - `"This is e.g. a classic example."`
     - `"That is i.e. the only explanation."`

4. **Academic Titles:**
   - **Input:** `"Dr. Smith and Prof. Jones analyzed the sample vs. the control group."`
   - **Expected Splits:**
     - `"Dr. Smith and Prof. Jones analyzed the sample vs. the control group."`

5. **Nested in Parentheses:**
   - **Input:** `"We studied the system (e.g. Fig. 1). This is a new sentence."`
   - **Expected Splits:**
     - `"We studied the system (e.g. Fig. 1)."`
     - `"This is a new sentence."`

You can run these test cases locally via the automated test suite:
```bash
conda run -n rag_prod python -m unittest test_sentence_splitter.py
```

### Can I use Agent 4 (interactive mode) instead of Agent 5 (batch mode)?

Yes. Agent 4 is ideal for testing individual sentences or when you want to see the LLM's reasoning:

```bash
python agent4_assistant.py --text "Topological insulators exhibit robust edge states." --top_k 5
```

It returns a suggested rewrite with citation and an explanation of why that source is relevant.

### What happens if I drop multiple PDFs at once?

The orchestrator uses a **30-second debounce timer** that resets with each new file. This means you can batch-drop as many files as you want — the pipeline only fires once, 30 seconds after the *last* file is dropped. If a pipeline run is already in progress, new files are queued for the next run.

---

## Data & Storage

### Where is the vector database stored?

In `physics_vectordb/` inside the project directory. This is a ChromaDB persistent store containing all text chunks, figure descriptions, and their `nomic-embed-text` embeddings.

### What is `bm25_index.pkl`?

A pickled `BM25Okapi` object that provides sparse keyword-based retrieval. It is automatically rebuilt from the entire ChromaDB collection every time Agent 3 or Agent 6 runs.

### Is the database incremental or does it rebuild every time?

**Incremental.** Agents 3 and 6 append new chunks to the existing ChromaDB collection — they never wipe and recreate it. Chunk IDs are sequentially numbered (`chunk_0`, `chunk_1`, …) and the ingestor reads the current max index before adding new entries.

### How do I reset / wipe the database?

Delete the `physics_vectordb/` directory and `bm25_index.pkl`:

```bash
rm -rf physics_vectordb/ bm25_index.pkl
```

Then re-run Agent 3 to rebuild from scratch.

### What file formats are supported for ingestion?

Only **PDF** files are supported. Agent 1 uses `pdftotext` for text extraction, and Agents 3/6 use PyMuPDF (`fitz`) + Detectron2 for layout-aware processing.

---

## Models & Performance

### How does Agent 3 handle figures and tables?

1. **Detectron2** (PubLayNet model) detects bounding boxes for text, figures, tables, titles, and lists on each page.
2. Figures and tables are **cropped** as PNG images and saved to `../extracted_data/images/`.
3. Each cropped image is passed to **`gemma4:latest`** (multimodal) with surrounding caption context, which generates a 3–5 sentence textual description.
4. These descriptions are embedded and stored in ChromaDB alongside the text chunks, making figures searchable by semantic content.

### How accurate is the citation matching?

Accuracy depends on:
- **Database coverage** — the system can only cite papers it has ingested. If a relevant paper wasn't downloaded or ingested, it won't be found.
- **Retrieval quality** — hybrid search (dense + sparse) generally outperforms either method alone.
- **LLM judgement** — `gemma4:latest` decides whether a sentence needs a citation and which retrieved source is most relevant.

Use the Ragas evaluation script (`evaluate_rag.py`) to quantitatively measure faithfulness and answer relevancy on your specific corpus.

### Can I swap `gemma4` for a different model?

Yes. Change the model name in `config.py`:

```python
# config.py
LLM_MODEL = "your-model:latest"     # Used by all agents and the supervisor
EMBED_MODEL = "your-embed-model"    # Used for dense vector search
EVAL_MODEL = "your-eval-model"      # Used by Ragas evaluation only
```

All agents import from `config.py`, so this single change propagates everywhere. For the supervisor, the model must support tool-calling. For figure description, a multimodal (vision-language) model is required.

### How long does the full pipeline take?

Rough estimates for a single source PDF with ~50 references:

| Stage | Time |
|-------|------|
| Agent 1 (extraction) | < 5 seconds |
| Agent 2 (fetching) | 3–10 minutes (network-bound, rate-limited) |
| Agent 3 (ingestion, 4 workers) | 10–30 minutes (GPU-bound, depends on page count) |
| Agent 5 (batch citing a 1-page draft) | 1–3 minutes |

---

## Evaluation

### How do I evaluate the RAG pipeline?

```bash
# Ensure the judge model is pulled
ollama pull deepseek-r1:14b

# Run evaluation
python evaluate_rag.py
```

The script reads sample queries from the `sample_inputs` file, runs the full RAG pipeline on each, and evaluates the outputs using the [Ragas](https://docs.ragas.io/) framework.

### What metrics are measured?

| Metric | What it measures |
|--------|------------------|
| **Faithfulness** | Is the generated answer grounded in the retrieved context? (i.e., no hallucination) |
| **Answer Relevancy** | Does the answer actually address the user's query? |

Results are saved per-query to `evaluation_results.csv`.

### Can I add my own evaluation queries?

Yes. Edit the `sample_inputs` file. Separate each query with a blank line:

```
First query text about quantum computing.

Second query about dark matter density.

Third query about graphene conductance.
```

---

## Troubleshooting

### Agent 3 crashes with CUDA out-of-memory (OOM)

Reduce the number of parallel workers:

```bash
python agent3_ingestor.py --workers 1
```

Each worker loads its own Detectron2 model instance, so GPU VRAM usage scales linearly with worker count.

### `pdftotext: command not found`

```bash
sudo apt-get install poppler-utils
```

### `ollama: connection refused`

The Ollama daemon must be running:

```bash
ollama serve
```

### Agent 4/5 says "collection not found"

Agent 3 must be run at least once to create the `physics_papers` ChromaDB collection before Agents 4 or 5 can query it.

### Agent 5 marks all sentences as "NO citation needed"

The LLM is being conservative. You can:
1. Make the prompt in `agent5_batch_citer.py`'s `needs_citation()` function more permissive.
2. Use Agent 4 interactively to cite specific sentences instead.

### The LangGraph supervisor loops or picks wrong tools

- Update to the latest `gemma4` weights: `ollama pull gemma4:latest`
- Bypass the supervisor entirely and run agents manually (see [Can I run agents individually?](#can-i-run-agents-individually-without-the-orchestrator)).

### Where are the logs?

All agents write to `logs/citation_agent.log` (rotating, 5 MB max, 3 backups). Console output is also logged. Check this file for detailed error traces:

```bash
tail -f logs/citation_agent.log
```

### arXiv rate limiting (HTTP 429)

Agent 2 enforces a 3-second sleep between arXiv requests. If you still hit limits, increase the `time.sleep(3)` values on lines 114 and 134 of `agent2_fetcher.py`.

### BM25 index shape mismatch

The BM25 index is rebuilt by Agent 3 from the full database. If you manually edit ChromaDB, re-run Agent 3 with `--workers 1` on an empty `downloaded.json` to force a rebuild.

---

## Advanced

### Can I use this for non-physics papers?

Yes. The pipeline is domain-agnostic — it works with any PDF that has a numbered reference section in the format `[N] ...`. The ChromaDB collection is named `physics_papers` by default, but you can change this in the configuration constants.

### Can I ingest papers from sources other than the reference list?

Yes, using Agent 6. Drop any PDF directly into `pulled_pdfs/` and it will be ingested into the database with a label derived from the filename (or a custom `--citation` string).

### How do I change the Unpaywall API email?

Edit `config.py`:

```python
UNPAYWALL_EMAIL = "your.email@example.com"
```

Using your own institutional email may improve rate limits from the Unpaywall API.

### Can I run the supervisor without the file watchers?

Yes. Import and call `process_event()` directly:

```python
from agent_graph import process_event
process_event("Please extract citations and fetch the papers.")
```

Or run it as a standalone script:

```bash
python agent_graph.py
```

### How do I contribute or extend the pipeline?

To add a new agent:
1. Create `agentN_your_agent.py` with a callable entry-point function.
2. Import shared utilities as needed (`from shared.search import hybrid_search`, `from config import LLM_MODEL`, etc.).
3. Define a `@tool`-decorated wrapper in `agent_graph.py` and add it to the `tools` list.
4. The LangGraph supervisor will automatically have access to your new tool.

---

*Last updated: 2026-05-17*
