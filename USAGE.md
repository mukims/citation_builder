# Citation Agent — User Guide

> **Who is this for?** Anyone who wants to automatically add citations to their
> writing using a local database of scientific papers.
> No coding knowledge required — just follow the steps below.

---

## Table of Contents

1. [What Does This System Do?](#1-what-does-this-system-do)
2. [Before You Begin — One-Time Setup](#2-before-you-begin--one-time-setup)
3. [Starting the System](#3-starting-the-system)
4. [Adding Papers to the Database](#4-adding-papers-to-the-database)
5. [Auto-Citing a Draft](#5-auto-citing-a-draft)
6. [Chatting With Your Papers](#6-chatting-with-your-papers)
7. [Understanding the Output Files](#7-understanding-the-output-files)
8. [Stopping the System](#8-stopping-the-system)
9. [Folder Map](#9-folder-map)
10. [Common Problems and Fixes](#10-common-problems-and-fixes)
11. [Advanced — Running Agents Individually](#11-advanced--running-agents-individually)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. What Does This System Do?

This tool does three things, all locally on your machine (no data leaves your computer):

| You do this | The system does this |
|---|---|
| Drop a **source PDF** into the `raw/` folder | Reads the reference list, downloads every open-access paper it finds, and stores them in a searchable database |
| Drop a **plain-text draft** (`.txt`) into the `drafts/` folder | Reads each sentence, decides which ones need citations, searches the database for matching papers, and rewrites those sentences with `\cite{key}` tags |
| Type a question in **chat mode** | Searches the database and gives you a conversational answer grounded in your papers |

Everything is automatic. Once the system is running, you just drop files into
the right folder.

---

## 2. Before You Begin — One-Time Setup

You only need to do this once.

### Step 1 — Open a terminal

On Ubuntu/Linux, press **Ctrl + Alt + T** to open a terminal window.

### Step 2 — Install system packages

Copy and paste this line into the terminal, then press Enter.
You may be asked for your password.

```
sudo apt-get install poppler-utils libgl1-mesa-glx libglib2.0-0
```

### Step 3 — Activate the environment

Every time you want to use the citation system, you need to activate the
`rag_prod` environment first:

```
conda activate rag_prod
```

Your terminal prompt should now show `(rag_prod)` at the beginning.

### Step 4 — Make sure Ollama is running

Ollama runs the AI models on your computer. Check that it is running:

```
ollama list
```

You should see a list of models. If you get "connection refused", start it:

```
ollama serve
```

### Step 5 — Pull the required AI models (first time only)

Run each line one at a time, waiting for each download to finish:

```
ollama pull gemma4:latest
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

These are large downloads (several GB each). Wait for all three to complete.

### Step 6 — Set your contact email for Unpaywall

Unpaywall (used in step 2 of the paper fetcher) requires a real contact address
on every request and rejects calls without one. Set it once in your shell:

```
export UNPAYWALL_EMAIL=you@institution.edu
```

Add that line to your `~/.bashrc` so it persists across terminals. If you skip
this, Agent 2 logs a warning and every lookup falls back to the arXiv search,
which finds far fewer papers.

**Setup complete!** From now on, you only need to run `conda activate rag_prod`
each time you open a new terminal.

---

## 3. Starting the System

Open a terminal and run:

```
conda activate rag_prod
cd ~/citation_builder
python master_orchestrator.py
```

You will see a banner when it is ready:

```
╔══════════════════════════════════════════════════════════════╗
║             Citation Agent — Master Orchestrator             ║
║  Watching:                                                   ║
║    • raw/          → extract → fetch → ingest                ║
║    • pulled_pdfs/  → auto-ingest into ChromaDB               ║
║    • drafts/       → auto-cite .txt files                    ║
╚══════════════════════════════════════════════════════════════╝
```

**Leave this terminal window open.** The system runs continuously until you
stop it.

To also start the **interactive research chat**, use:

```
python master_orchestrator.py --chat
```

> **First startup:** The system may spend a few minutes syncing the database.
> This is normal — it only processes papers it has not seen before.

---

## 4. Adding Papers to the Database

### Method A — From a source PDF's reference list (recommended)

Use this when you have a paper and want to automatically download everything
it cites.

1. Make sure the system is running (see above).
2. Open your file manager and go to `~/citation_builder/raw/`.
3. **Copy or drag** your PDF into this folder.
4. Watch the terminal — the system will:
   - Extract every reference from the paper
   - Search Crossref, Unpaywall, and arXiv for each one
   - Download every open-access paper it can find
   - Process and store them in the database

> **Dropping multiple files?** Go ahead! The system waits 30 seconds after the
> last file before starting, so you can drop a whole batch at once.

### Method B — Adding a specific PDF directly

If you already have a PDF you want in the database:

1. Copy the PDF into `~/citation_builder/pulled_pdfs/`.
2. Done. The system ingests it automatically within a few seconds.

---

## 5. Auto-Citing a Draft

1. **Write your draft** as a plain text file (`.txt`). Use any text editor.
2. **Save or copy** it into `~/citation_builder/drafts/`.
   - Example: `drafts/my_introduction.txt`
3. The system detects the file and processes it. You will see progress:
   ```
   [1/25] The electronic properties of graphene nanoribbons...
    -> Needs citation. Searching context…
    -> Cited: The electronic properties... \cite{cite_1}
   ```
4. When finished, **three output files** appear in `drafts/`:

| Output file | What is inside |
|-------------|----------------|
| `my_introduction_cited.txt` | Your draft with `\cite{cite_1}` etc. inserted |
| `my_introduction_citations.json` | Which `cite_N` maps to which full reference |
| `my_introduction_report.md` | Why each citation was chosen (sentence by sentence) |

**Important:**
- Your file must end in `.txt` (not `.docx`, `.pdf`, or `.tex`).
- Files ending in `_cited.txt` are ignored (to avoid re-processing outputs).
- Only factual-claim sentences are cited — headings and opinions are left alone.

---

## 6. Chatting With Your Papers

Chat mode lets you ask questions about your paper database — like having a
conversation with your entire literature collection. It uses a fast model
(`qwen2.5:7b`) with streaming, so answers appear word-by-word.

### Starting a chat

```
conda activate rag_prod
cd ~/citation_builder
python master_orchestrator.py --chat
```

Wait for the `You:` prompt, then type your question:

```
You: What are the main approaches to Green's functions in nanoribbons?

Assistant: Based on the papers in your database, there are three primary
approaches... [streams word-by-word]

  Sources: recursive_greens_function_59, electronic_transport_22
  (type /sources for full list)
```

### Chat commands

| Command | What it does |
|---------|--------------|
| `/clear` | Clears conversation history (start fresh) |
| `/sources` | Shows which papers were used in the last answer |
| `/export` | Saves the whole conversation to a file in `drafts/` |
| `/help` | Shows all available commands |
| `quit` or `exit` | Ends the chat session |

> **Tip:** You can ask follow-up questions. The assistant remembers the last
> few turns of conversation so you can refine your queries.

---

## 7. Understanding the Output Files

After citing a draft, you get three files. Here is what to do with each:

### The cited draft (`_cited.txt`)

This is your original text with LaTeX citation commands inserted. For example:

```
Graphene nanoribbons exhibit quantum confinement effects \cite{cite_1}.
```

Copy these sentences into your LaTeX document.

### The citation mapping (`_citations.json`)

This tells you which reference each `cite_N` refers to:

```json
{
    "cite_1": "K. Wakabayashi et al., Electronic transport in graphene...",
    "cite_2": "Y.-W. Son et al., Energy gaps in graphene nanoribbons..."
}
```

Use this to build your bibliography / BibTeX entries.

### The citation report (`_report.md`)

A Markdown document you can open in any text editor or preview tool. It shows:
- How many sentences were cited vs. skipped
- For each cited sentence: the original text, the rewritten version, and a
  2-3 sentence explanation of why that citation was chosen.

---

## 8. Stopping the System

Press **Ctrl + C** in the terminal where the system is running. You will see:

```
Shutting down...
Orchestrator stopped. Goodbye.
```

All pending work is cancelled cleanly.

---

## 9. Folder Map

Here is what each folder and important file does:

```
~/citation_builder/
|
|-- raw/                  <-- DROP source PDFs here (the system reads their
|   |                         reference lists)
|   |-- processed/        Source PDFs whose references were read successfully
|   `-- failed/           Source PDFs that yielded no references — usually a
|                             scan with no text layer, or a reference list in
|                             a format the patterns do not recognise
|
|-- pulled_pdfs/          <-- DROP individual PDFs here to add them directly
|                             (also where auto-downloaded papers go)
|
|-- drafts/               <-- DROP your .txt drafts here for auto-citation
|                             (output files also appear here)
|
|-- physics_vectordb/     The searchable database (created automatically)
|-- bm25_index.pkl        Text search index (created automatically)
|-- images/               Figure crops from ingestion. Written, described by
|                             the vision model, then never read again — safe
|                             to delete whenever you like
|-- logs/                 Log files (check here if something goes wrong)
|-- attic/                Kept for reference, wired into nothing
|-- config.py             Settings file (model names, timing, etc.)
`-- prompts.py            Every prompt sent to a model
```

---

## 10. Common Problems and Fixes

### "conda: command not found"

Conda is not installed or not in your path. Follow the
[Miniconda install guide](https://docs.conda.io/en/latest/miniconda.html),
then try again.

### "(rag_prod) does not appear in my prompt"

Run `conda activate rag_prod` again. If it says the environment does not exist,
you may need to create it first — ask your system administrator or refer to the
project's `environment.yml`:

```
conda env create -f environment.yml
```

### "ollama: connection refused"

The Ollama service is not running. Start it:

```
ollama serve
```

Then try again in a second terminal window.

### "pdftotext: command not found"

```
sudo apt-get install poppler-utils
```

### "No module named watchdog" (or chromadb, ollama, etc.)

A Python package is missing. Activate the environment and install it:

```
conda activate rag_prod
python -m pip install -r requirements.txt
```

Replace `watchdog` with whatever module name is mentioned in the error.

### The system says "collection not found" when citing a draft

You need papers in the database first. Drop at least one PDF into `raw/` or
`pulled_pdfs/` and wait for ingestion to complete before trying to cite.

### Chat says `ResearchChat object has no attribute`

You need to restart the system. Press Ctrl+C, then run
`python master_orchestrator.py --chat` again.

### Processing seems stuck / very slow

- PDF ingestion involves running an AI model over every page. On a machine
  without a GPU, this can take 1-3 minutes per paper. This is normal.
- The chat uses a lightweight model (`qwen2.5:7b`) and streams output,
  so you should see words appearing within a few seconds.
- Check `logs/citation_agent.log` for detailed progress:
  ```
  tail -f logs/citation_agent.log
  ```

---

## 11. Advanced — Running Agents Individually

Most users should use the orchestrator (Section 3). But if you prefer
step-by-step control, you can run each agent as a standalone script.

**Always activate the environment first:**

```
conda activate rag_prod
cd ~/citation_builder
```

| What you want to do | Command |
|---------------------|---------|
| Extract references from PDFs in `raw/` | `python agent1_extractor.py` |
| Download the extracted references | `python agent2_fetcher.py` |
| Ingest downloaded papers into the database | `python agent3_ingestor.py` |
| Get a citation for a single sentence | `python agent4_assistant.py --text "Your sentence here."` |
| Cite an entire draft file | `python agent5_batch_citer.py --file drafts/my_draft.txt` |
| Ingest one PDF you added by hand | `python agent6_manual_ingestor.py --once pulled_pdfs/paper.pdf` |
| Start the research chat on its own | `python agent7_research_chat.py` |

Run them in order: Agent 1 first, then 2, then 3. After that, Agents 4 and 5
can be used in any order.

---

## 12. Configuration Reference

All settings live in one file: **`config.py`**. You usually do not need to
change anything, but here are the key settings if you want to tweak behaviour:

### AI Models

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `LLM_MODEL` | `gemma4:latest` | Main model for citation tasks and ingestion |
| `CHAT_MODEL` | `qwen2.5:7b` | Lightweight model for the interactive chat |
| `EMBED_MODEL` | `nomic-embed-text` | Text embedding model for search |
| `EVAL_MODEL` | `deepseek-r1:14b` | Evaluation model (only used for testing) |

### Chat Performance

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `CHAT_OLLAMA_OPTIONS["num_ctx"]` | `4096` | Context window size (smaller = faster) |
| `CHAT_OLLAMA_OPTIONS["num_thread"]` | `16` | CPU threads for chat model |
| `CHAT_OLLAMA_OPTIONS["flash_attn"]` | `True` | Flash Attention (reduces memory, faster) |
| `CHAT_OLLAMA_OPTIONS["kv_cache_type"]` | `"q8_0"` | 8-bit quantized KV cache (lower latency) |

### Timing

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `PDF_COOLDOWN_SECONDS` | `30` | How long to wait after the last PDF drop before processing |
| `DRAFT_COOLDOWN_SECONDS` | `2` | How quickly drafts are picked up after saving |

### Search

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `DEFAULT_TOP_K` | `3` | Number of matching paper excerpts returned per search |

### Environment Variables

Some settings are read from your shell environment rather than hardcoded, so
that personal details never end up committed to the repository.

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `UNPAYWALL_EMAIL` | `your-email@example.com` | Contact address sent to the Unpaywall API. Unpaywall rejects requests without a real address, so leaving this unset makes Agent 2 skip straight to the arXiv fallback. See Step 6 of the setup. |
| `CITATION_IMAGES_DIR` | `images/` | Where figure crops go during ingestion. Each crop is described by the vision model and then never read again, so this folder is a debugging aid — delete it whenever you like. |
| `CITATION_LOG_DIR` | `logs/` | Where the rotating log file is written. |
| `CITATION_LOG_FILE` | `1` | Set to `0` to log to the console only and leave no file behind. |
