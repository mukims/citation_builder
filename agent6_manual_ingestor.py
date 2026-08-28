"""
Agent 6 — Manual PDF Ingestor (agent6_manual_ingestor.py)

Monitors the `pulled_pdfs/` directory for manually dropped PDF files.
When a new PDF is detected, it processes the file directly through the full
multimodal ingestion pipeline (Detectron2 layout detection → gemma4 VLM figure
description → ChromaDB + BM25 indexing) without requiring a citation string from
downloaded.json.

This is the companion to Agent 3 for cases where Agent 2 could not automatically
find and download a paper (e.g., paywalled papers that you have downloaded manually).

Usage (standalone):
    python agent6_manual_ingestor.py                 # watch pulled_pdfs/ continuously
    python agent6_manual_ingestor.py --once <file>   # ingest a single PDF directly

The script can also be imported and called programmatically:
    from agent6_manual_ingestor import ingest_manual_pdf
    ingest_manual_pdf("pulled_pdfs/mypaper.pdf", citation_string="Smith et al. 2024")
"""

import os
import sys
import time
import threading
import argparse

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Error: 'watchdog' is not installed. Run: pip install watchdog")
    sys.exit(1)

from config import PULLED_PDFS_DIR, MANUAL_COOLDOWN_SECONDS
from shared.log import get_logger
from shared.ingestion import ingest_pdfs

logger = get_logger("agent6")


# ─── Core Ingestion Logic ──────────────────────────────────────────────────────


def ingest_manual_pdf(pdf_path: str, citation_string: str | None = None, workers: int = 1):
    """
    Ingest a single manually placed PDF into the ChromaDB vector database.

    Delegates to the shared ingestion path, so a PDF dropped here is subject to
    the same already-ingested check and manifest bookkeeping as one that arrives
    via Agent 3 — re-dropping a paper no longer re-runs layout detection and the
    VLM over every page of it.

    Args:
        pdf_path:        Absolute or relative path to the PDF file.
        citation_string: Optional citation/reference label to tag the document with.
                         If None, the PDF filename (without extension) is used.
        workers:         Worker processes for the parsing stage (single file, so
                         this is only useful if the file is very large).

    Returns:
        dict: The result summary from :func:`shared.ingestion.ingest_pdfs`.
    """
    if citation_string is None:
        citation_string = os.path.splitext(os.path.basename(pdf_path))[0]

    logger.info("Ingesting %s (citation label: '%s')", pdf_path, citation_string)
    return ingest_pdfs({pdf_path: citation_string}, workers=workers)


# ─── Watchdog Handler ─────────────────────────────────────────────────────────


class ManualPDFHandler(FileSystemEventHandler):
    """Debounced watchdog handler for the pulled_pdfs/ directory."""

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_created(self, event):
        self._schedule(event)

    def on_moved(self, event):
        # Handles files moved/renamed into the directory
        self._schedule(event, use_dest=True)

    def _schedule(self, event, use_dest=False):
        path = getattr(event, "dest_path", None) if use_dest else event.src_path
        if not path or event.is_directory or not path.lower().endswith(".pdf"):
            return
        with self._lock:
            if path in self._timers:
                self._timers[path].cancel()
            timer = threading.Timer(MANUAL_COOLDOWN_SECONDS, self._process, args=[path])
            self._timers[path] = timer
            timer.start()
            logger.info(
                "PDF detected: %s — processing in %ds…",
                os.path.basename(path),
                MANUAL_COOLDOWN_SECONDS,
            )

    def _process(self, path):
        with self._lock:
            self._timers.pop(path, None)
        ingest_manual_pdf(path)


# ─── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Agent 6 — Manual PDF Ingestor. Watches pulled_pdfs/ for new files."
    )
    parser.add_argument(
        "--once",
        metavar="PDF_PATH",
        help="Ingest a single PDF immediately instead of watching the directory.",
    )
    parser.add_argument(
        "--citation",
        metavar="CITATION_STRING",
        default=None,
        help="Optional citation label for --once mode. Defaults to the PDF filename.",
    )
    args = parser.parse_args()

    if args.once:
        ingest_manual_pdf(args.once, citation_string=args.citation)
        return

    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)
    handler = ManualPDFHandler()
    observer = Observer()
    observer.schedule(handler, path=PULLED_PDFS_DIR, recursive=False)
    observer.start()

    logger.info("Watching '%s/' for manually placed PDFs…", PULLED_PDFS_DIR)
    logger.info("Drop any PDF into the folder to auto-ingest it into ChromaDB.")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watcher…")
        with handler._lock:
            for t in handler._timers.values():
                t.cancel()
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
