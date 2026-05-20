# Citation Agent — User Guide

> **Who is this for?** Anyone who wants to automatically add citations to their writing using a local database of scientific papers. No coding knowledge required — just follow the steps below.

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
9. [Folder Map — Where Things Live](#9-folder-map--where-things-live)
10. [Common Problems & Fixes](#10-common-problems--fixes)
11. [Advanced: Running Agents Individually](#11-advanced-running-agents-individually)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. What Does This System Do?

This tool does three things, all locally on your machine (no data leaves your computer):

| You do this… | …and the system does this |
|---|---|
| Drop a **source PDF** into the `raw/` folder | Reads the reference list → downloads every open-access paper it finds → stores them in a searchable database |
| Drop a **plain-text draft** (`.txt`) into the `drafts/` folder | Reads each sentence → decides which ones need citations → searches the database for matching papers → rewrites those sentences with `\cite{key}` tags and produces a mapping file |
| Type a question in **chat mode** | Searches the database for relevant papers and gives you a grounded, conversational answer with sources |

Everything is automatic. Once the system is running, you just drop files into the right folder.

---

## 2. Before You Begin — One-Time Setup

You only need to do this once.

### Step 1: Open a terminal

On Ubuntu/Linux, press `Ctrl + Alt + T` to open a terminal window.

### Step 2: Install system packages

Copy and paste the following line into the terminal, then press Enter. You may be asked for your password.

```
sudo apt-get install poppler-utils libgl1-mesa-glx libglib2.0-0
```

### Step 3: Activate the environment

Every time you want to use the citation system, you must first activate the `rag_prod` environment. Copy and paste:

```
conda activate rag_prod
```

Your terminal prompt should now show `(rag_prod)` at the beginning. If it doesn't, something went wrong — see [Common Problems](#10-common-problems--fixes).

### Step 4: Make sure Ollama is running

Ollama is the program that runs the AI models locally. It should already be installed. Check that it's running:

```
ollama list
```

You should see a list of models. If you get `connection refused`, start Ollama first:

```
ollama serve
```

### Step 5: Pull the required AI models (first time only)

```
ollama pull gemma4:latest
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

Each model is a large download (several GB). Wait for each to finish before running the next.

### ✅ Setup complete!

You won't need to repeat these steps. From now on, you only need Step 3 (activating the environment) each time you open a new terminal.

---

## 3. Starting the System

### Step 1: Open a terminal and activate the environment

```
conda activate rag_prod
```

### Step 2: Navigate to the project folder

```
cd ~/citation_builder
```

### Step 3: Start the system

**Basic mode** — watches folders and processes files automatically:

```
python master_orchestrator.py
```

**Chat mode** — same as above, plus an interactive research assistant:

```
python master_orchestrator.py --chat
```

You'll see a banner like this when it's ready:

```
╔══════════════════════════════════════════════════════════════╗
║             Citation Agent — Master Orchestrator             ║
╠══════════════════════════════════════════════════════════════╣
║  Watching:                                                   ║
║    • raw/          → extract → fetch → ingest                ║
║    • pulled_pdfs/  → auto-ingest into ChromaDB               ║
║    • drafts/       → auto-cite .txt files                    ║
╚══════════════════════════════════════════════════════════════╝
```

**Leave this terminal window open.** The system runs continuously until you stop it.

> **⏱ Tip:** On the first startup (or if there are new unprocessed papers), the system may spend several minutes syncing the database. This is normal — it only processes papers it hasn't seen before.

---

## 4. Adding Papers to the Database

There are two ways to add papers:

### Method A: From a source PDF's reference list (recommended)

Use this when you have a paper and want to automatically download everything it cites.

1. Make sure the system is running (see [Starting the System](#3-starting-the-system)).
2. Open your file manager and navigate to `~/citation_builder/raw/`.
3. Copy your source PDF into this folder.
4. The system will detect the file within seconds. In the terminal, you'll see it start extracting citations.
5. It will then automatically:
   - Extract every reference from the paper
   - Search Crossref, Unpaywall, and arXiv for each reference
   - Download every open-access paper it can find
   - Process and store them in the database

> **Batch drop:** You can drop multiple PDFs at once. The system waits 30 seconds after the last file before starting — this lets you drop a whole batch without triggering the pipeline for each file individually.

### Method B: Adding a specific PDF directly

Use this when you already have a PDF file and want to add it to the database.

1. Copy the PDF into `~/citation_builder/pulled_pdfs/`.
2. The system will detect it and ingest it automatically (within ~5 seconds).

---

## 5. Auto-Citing a Draft

1. Write your draft as a **plain text file** (`.txt`). You can use any text editor (e.g. gedit, VS Code, Notepad).
2. Save it (or copy it) into `~/citation_builder/drafts/`. For example: `drafts/my_introduction.txt`.
3. The system detects the file and starts processing it. In the terminal you'll see progress like:
   ```
   [1/25] The electronic properties of graphene nanoribbons...
    -> Needs citation. Searching context…
    -> Cited: The electronic properties of graphene nanoribbons \cite{cite_1}...
   ```
4. When it finishes, three output files appear in the `drafts/` folder:

| File | What it contains |
|------|-----------------|
| `my_introduction_cited.txt` | Your draft with `\cite{cite_1}`, `\cite{cite_2}`, etc. inserted |
| `my_introduction_citations.json` | A mapping of each `cite_N` key to the full reference string |
| `my_introduction_report.md` | A detailed report explaining *why* each citation was chosen |

> **Important:** Your draft must be a `.txt` file (not `.docx`, `.pdf`, or `.tex`). Files ending in `_cited.txt` are ignored to avoid re-processing output files.

---

## 6. Chatting With Your Papers

Chat mode lets you ask questions about the papers in your database — like having a conversation with your literature collection.

### Starting a chat session

```
python master_orchestrator.py --chat
```

Once loaded, you'll see a `You:` prompt. Just type your question:

```
You: What are the main approaches to calculating Green's functions in nanoribbons?
