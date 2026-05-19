"""
Master Orchestrator — The single entrypoint for the Citation Agent pipeline.

    python master_orchestrator.py             # watch mode (default)
    python master_orchestrator.py --chat       # watch mode + interactive research chat

On startup the orchestrator:
  1. Ensures all directories exist (raw/, drafts/, pulled_pdfs/)
  2. Syncs the database — any PDFs in pulled_pdfs/ that aren't in ChromaDB
     are automatically ingested before the watchers start.
  3. Starts three file watchers:
       • raw/         → extracts citations → fetches papers → ingests them
       • pulled_pdfs/ → ingests new PDFs directly
       • drafts/      → auto-cites new .txt drafts
  4. (Optional) Opens an interactive research chat in the foreground.

Users never need to run individual agent scripts — just drop files into the
right directories and the orchestrator handles the rest.
"""

import os
import sys
import glob
import time
import threading
import argparse
import concurrent.futures

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Error: 'watchdog' package is not installed. Please run: pip install watchdog")
    sys.exit(1)

from config import (
    RAW_DIR,
    DRAFTS_DIR,
    PULLED_PDFS_DIR,
    PDF_COOLDOWN_SECONDS,
    DRAFT_COOLDOWN_SECONDS,
    MANUAL_COOLDOWN_SECONDS,
    DEFAULT_WORKERS,
    VECTORDB_PATH,
    COLLECTION_NAME,
)
from shared.log import get_logger
from shared.ingestion import process_pdf, upsert_corpus, rebuild_bm25, get_ingested_documents, mark_document_ingested

logger = get_logger("orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Startup Database Sync
# ═══════════════════════════════════════════════════════════════════════════════


def sync_database(workers=1):
    """Ensure every PDF in pulled_pdfs/ is indexed in ChromaDB.

    Called once at startup so the system is always in a consistent state,
    even if Agent 3 was interrupted or new files were placed while the
    orchestrator was offline.
    """
    pdf_files = glob.glob(os.path.join(PULLED_PDFS_DIR, "*.pdf"))
    if not pdf_files:
        logger.info("[Sync] No PDFs in pulled_pdfs/ — nothing to sync.")
        return

    # Determine which PDFs are already ingested
    already_ingested = get_ingested_documents()
    missing = []
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path).strip().replace(" ", "_").lower()
        if pdf_name not in already_ingested:
            missing.append(pdf_path)

    if not missing:
        logger.info(
            "[Sync] All %d PDFs in pulled_pdfs/ are already in the database. ✓",
            len(pdf_files),
        )
        return

    logger.info(
        "[Sync] Found %d/%d PDFs not yet in the database. Ingesting…",
        len(missing), len(pdf_files),
    )

    corpus = []
    if workers <= 1:
        for i, pdf_path in enumerate(missing):
            citation_label = os.path.splitext(os.path.basename(pdf_path))[0]
            logger.info("[Sync] [%d/%d] Processing %s…", i + 1, len(missing), pdf_path)
            corpus.extend(process_pdf(pdf_path, citation_label))
    else:
        logger.info("[Sync] Processing %d PDFs using %d workers…", len(missing), workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for pdf_path in missing:
                citation_label = os.path.splitext(os.path.basename(pdf_path))[0]
                futures.append(executor.submit(process_pdf, pdf_path, citation_label))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    corpus.extend(future.result())
                except Exception as e:
                    logger.error("[Sync] Worker failed: %s", e)

    if corpus:
        inserted = upsert_corpus(corpus)
        logger.info("[Sync] Inserted %d new chunks.", inserted)
        rebuild_bm25()
        logger.info("[Sync] Database sync complete. ✓")
    else:
        logger.info("[Sync] No content extracted from new PDFs.")

    # Mark all attempted PDFs as ingested so we don't retry duplicates/failures forever
    for pdf_path in missing:
        pdf_name = os.path.basename(pdf_path).strip().replace(" ", "_").lower()
        mark_document_ingested(pdf_name)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. File Watchers
# ═══════════════════════════════════════════════════════════════════════════════


class RawPDFHandler(FileSystemEventHandler):
    """Watches raw/ for new source PDFs → triggers the full pipeline."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            logger.info("[raw/] New PDF detected: %s", os.path.basename(event.src_path))
            self.orchestrator.trigger_pdf_cooldown()


class PulledPDFHandler(FileSystemEventHandler):
    """Watches pulled_pdfs/ for manually placed or newly fetched PDFs → ingests them."""

    def __init__(self):
        self._timers = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        self._schedule(event)

    def on_moved(self, event):
        self._schedule(event, use_dest=True)

    def _schedule(self, event, use_dest=False):
        path = getattr(event, "dest_path", None) if use_dest else event.src_path
        if not path or event.is_directory or not path.lower().endswith(".pdf"):
            return
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(MANUAL_COOLDOWN_SECONDS, self._ingest, args=[path])
            self._timers[path] = timer
            timer.start()
            logger.info(
                "[pulled_pdfs/] PDF detected: %s — ingesting in %ds…",
                os.path.basename(path), MANUAL_COOLDOWN_SECONDS,
            )

    def _ingest(self, path):
        with self._lock:
            self._timers.pop(path, None)
        pdf_name = os.path.basename(path).strip().replace(" ", "_").lower()
        try:
            citation_label = os.path.splitext(os.path.basename(path))[0]
            corpus = process_pdf(path, citation_label)
            if corpus:
                inserted = upsert_corpus(corpus)
                rebuild_bm25()
                logger.info("[pulled_pdfs/] ✓ Ingested %s (%d chunks).", os.path.basename(path), inserted)
            else:
                logger.warning("[pulled_pdfs/] No content extracted from %s.", os.path.basename(path))
            # Always mark as ingested to prevent retry loops on duplicates
            mark_document_ingested(pdf_name)
        except Exception as e:
            logger.error("[pulled_pdfs/] Failed to ingest %s: %s", path, e)
            mark_document_ingested(pdf_name)


class DraftHandler(FileSystemEventHandler):
    """Watches drafts/ for new or modified .txt files → triggers Agent 5."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def on_created(self, event):
        self._handle(event)

    def on_modified(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        path = event.src_path.lower()
        if path.endswith(".txt") and not path.endswith("_cited.txt"):
            logger.info("[drafts/] Draft update detected: %s", os.path.basename(event.src_path))
            self.orchestrator.trigger_draft_run(event.src_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Orchestrator Core
# ═══════════════════════════════════════════════════════════════════════════════


class Orchestrator:
    def __init__(self, workers=1):
        self.workers = workers
        self.pdf_timer = None
        self.pdf_lock = threading.Lock()
        self.is_processing_pdf = False
        self.pdf_run_pending = False

        self.draft_timers = {}
        self.draft_lock = threading.Lock()

    # ── Raw PDF pipeline (Agent 1 → 2 → 3) ──────────────────────────────

    def trigger_pdf_cooldown(self):
        with self.pdf_lock:
            if self.is_processing_pdf:
                logger.info("[raw/] Pipeline currently running. File queued for NEXT run.")
                self.pdf_run_pending = True
            else:
                self._schedule_pdf_run()

    def _schedule_pdf_run(self):
        if self.pdf_timer:
            self.pdf_timer.cancel()
        logger.info(
            "[raw/] Pipeline scheduled in %ds… (drop more files to reset timer)",
            PDF_COOLDOWN_SECONDS,
        )
        self.pdf_timer = threading.Timer(PDF_COOLDOWN_SECONDS, self._run_pipeline)
        self.pdf_timer.start()

    def _run_pipeline(self):
        with self.pdf_lock:
            self.is_processing_pdf = True
            self.pdf_run_pending = False

        try:
            # Lazy import to avoid loading LangGraph at startup
            from agent_graph import process_event
            logger.info("Notifying Supervisor Agent of new PDFs…")
            process_event(
                f"New PDFs have been added to the raw/ directory. "
                f"Please extract citations from them, fetch the papers, and ingest them using {self.workers} workers."
            )
        except Exception as e:
            logger.error("Pipeline error: %s", e)
        finally:
            with self.pdf_lock:
                self.is_processing_pdf = False
                if self.pdf_run_pending:
                    logger.info("[raw/] Resolving pending queued PDFs…")
                    self._schedule_pdf_run()

    # ── Draft citation (Agent 5) ─────────────────────────────────────────

    def trigger_draft_run(self, filepath):
        with self.draft_lock:
            if filepath in self.draft_timers:
                self.draft_timers[filepath].cancel()
            timer = threading.Timer(DRAFT_COOLDOWN_SECONDS, self._run_citer, args=[filepath])
            self.draft_timers[filepath] = timer
            timer.start()

    def _run_citer(self, filepath):
        try:
            from agent_graph import process_event
            logger.info("Notifying Supervisor Agent of new draft: %s…", filepath)
            process_event(
                f"A new draft text file needs citation processing. "
                f"The file is located at: {filepath}. Please use the batch cite tool."
            )
        except Exception as e:
            logger.error("Draft citing error: %s", e)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def shutdown(self):
        if self.pdf_timer:
            self.pdf_timer.cancel()
        for timer in self.draft_timers.values():
            timer.cancel()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║             Citation Agent — Master Orchestrator             ║
╠══════════════════════════════════════════════════════════════╣
║  Watching:                                                   ║
║    • raw/          → extract → fetch → ingest                ║
║    • pulled_pdfs/  → auto-ingest into ChromaDB               ║
║    • drafts/       → auto-cite .txt files                    ║
║                                                              ║
║  Drop files into the right folder and the system handles     ║
║  everything automatically.                                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="Citation Agent — Master Orchestrator. Single entrypoint for the entire pipeline.",
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="After startup, open an interactive research chat session (Agent 7).",
    )
    parser.add_argument(
        "--no-sync", action="store_true",
        help="Skip the startup database sync (faster startup if you know the DB is current).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel workers for ingestion (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()

    # ── Ensure directories exist ─────────────────────────────────────────
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)

    # ── Step 1: Sync database ────────────────────────────────────────────
    if not args.no_sync:
        logger.info("Step 1/2: Syncing database using %d workers…", args.workers)
        sync_database(workers=args.workers)
    else:
        logger.info("Step 1/2: Database sync skipped (--no-sync).")

    # ── Step 2: Start watchers ───────────────────────────────────────────
    logger.info("Step 2/2: Starting file watchers…")
    orchestrator = Orchestrator(workers=args.workers)
    observer = Observer()

    observer.schedule(RawPDFHandler(orchestrator), path=RAW_DIR, recursive=False)
    observer.schedule(PulledPDFHandler(), path=PULLED_PDFS_DIR, recursive=False)
    observer.schedule(DraftHandler(orchestrator), path=DRAFTS_DIR, recursive=False)

    observer.start()
    print(BANNER)
    logger.info("All watchers active. Drop files into the folders above.")

    # ── Step 3: Process existing raw PDFs ────────────────────────────────
    raw_pdfs = [f for f in glob.glob(os.path.join(RAW_DIR, "*.pdf")) if os.path.isfile(f)]
    if raw_pdfs:
        logger.info("Found %d unprocessed PDFs in raw/ on startup. Triggering pipeline…", len(raw_pdfs))
        orchestrator.trigger_pdf_cooldown()

    # ── Foreground mode ──────────────────────────────────────────────────
    try:
        if args.chat:
            # Launch interactive research chat in the foreground
            # (watchers continue running in background threads)
            from agent7_research_chat import ResearchChat

            print("  🔬  Research Chat is loading…\n")
            try:
                agent = ResearchChat(top_k=5)
            except RuntimeError:
                print("\n" + "=" * 60)
                print("  ⚠  No paper database found yet.")
                print("")
                print("  The research chat requires ingested papers to search.")
                print("  Drop PDFs into raw/ or pulled_pdfs/ first — the")
                print("  orchestrator will ingest them automatically.")
                print("")
                print("  Once papers are ingested, restart with --chat.")
                print("=" * 60 + "\n")
                logger.info("Falling back to watch mode (no database for chat).")
                args.chat = False  # fall through to watch mode below

        if args.chat:
            print("=" * 60)
            print("  Type your research questions below.")
            print("  File watchers are running in the background.")
            print("  Type /help for commands, 'quit' to exit.")
            print("=" * 60 + "\n")

            from agent7_research_chat import HELP_TEXT
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nShutting down…")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break
                if user_input == "/help":
                    print(HELP_TEXT)
                    continue
                if user_input == "/clear":
                    agent.clear_history()
                    print("✓ Conversation history cleared.\n")
                    continue
                if user_input == "/sources":
                    if not agent.last_sources:
                        print("No sources from the last response.\n")
                    else:
                        print("\n📚 Sources used in the last response:")
                        for i, s in enumerate(agent.last_sources, 1):
                            print(f"  {i}. [{s['document']}] {s['citation']}")
                            print(f"     Relevance: {s['relevance_score']}")
                        print()
                    continue
                if user_input == "/export":
                    path = agent.export_conversation()
                    print(f"✓ Conversation exported to {path}\n")
                    continue

                try:
                    print("\nAssistant: ", end="", flush=True)
                    for chunk in agent.chat_stream(user_input):
                        print(chunk, end="", flush=True)
                    print("\n")
                    if agent.last_sources:
                        cits = {s["citation"][:60] for s in agent.last_sources[:3]}
                        print(f"  📚 Drawing from: {', '.join(cits)}")
                        print("  (type /sources for full list)\n")
                except Exception as e:
                    logger.error("Chat error: %s", e)
                    print(f"\n⚠ Error: {e}. Please try again.\n")
        else:
            # Headless watch mode — just idle until Ctrl+C
            logger.info("Running in watch mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        orchestrator.shutdown()
        observer.stop()
        observer.join()
        logger.info("Orchestrator stopped. Goodbye.")


if __name__ == "__main__":
    main()
