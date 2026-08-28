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
)
from shared.log import get_logger
from shared.ingestion import ingest_pdfs

logger = get_logger("orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Startup Database Sync
# ═══════════════════════════════════════════════════════════════════════════════


def sync_database(workers=1):
    """Ensure every PDF in pulled_pdfs/ is indexed in ChromaDB.

    Called once at startup so the system is always in a consistent state, even
    if ingestion was interrupted or files were placed while the orchestrator
    was offline.
    """
    pdf_files = glob.glob(os.path.join(PULLED_PDFS_DIR, "*.pdf"))
    if not pdf_files:
        logger.info("[Sync] No PDFs in pulled_pdfs/ — nothing to sync.")
        return None

    result = ingest_pdfs(pdf_files, workers=workers, log_prefix="[Sync] ")
    logger.info(
        "[Sync] Complete — %d processed, %d already indexed, %d new chunk(s). ✓",
        result["processed"], result["skipped"], result["inserted"],
    )
    return result


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
    """Watches pulled_pdfs/ for fetched or manually placed PDFs and ingests them.

    Arrivals are collected into a pending set behind a single debounce timer
    that resets on each new file, then ingested as one batch. Agent 2 writes its
    downloads straight into this directory, so a fetch run can drop dozens of
    files at once; batching means the BM25 index is rebuilt once for the run
    rather than once per paper (a full rebuild re-tokenises the entire
    collection, so per-file rebuilds get quadratically expensive).
    """

    def __init__(self, workers=1):
        self.workers = workers
        self._pending = set()
        self._timer = None
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
            self._pending.add(path)
            pending = len(self._pending)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(MANUAL_COOLDOWN_SECONDS, self._drain)
            self._timer.start()
        logger.info(
            "[pulled_pdfs/] %d file(s) queued — ingesting in %ds…",
            pending, MANUAL_COOLDOWN_SECONDS,
        )

    def _drain(self):
        with self._lock:
            batch = sorted(self._pending)
            self._pending.clear()
            self._timer = None
        if not batch:
            return
        try:
            ingest_pdfs(batch, workers=self.workers, log_prefix="[pulled_pdfs/] ")
        except Exception as e:
            logger.error("[pulled_pdfs/] Batch ingest failed: %s", e)

    def shutdown(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None


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

    pulled_handler = PulledPDFHandler(workers=args.workers)
    observer.schedule(RawPDFHandler(orchestrator), path=RAW_DIR, recursive=False)
    observer.schedule(pulled_handler, path=PULLED_PDFS_DIR, recursive=False)
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
        pulled_handler.shutdown()
        observer.stop()
        observer.join()
        logger.info("Orchestrator stopped. Goodbye.")


if __name__ == "__main__":
    main()
