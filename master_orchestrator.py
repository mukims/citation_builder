import os
import time
import threading
import sys
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
    DEFAULT_WORKERS,
)
from agent_graph import process_event
from agent6_manual_ingestor import ManualPDFHandler
from shared.log import get_logger

logger = get_logger("orchestrator")


class PDFHandler(FileSystemEventHandler):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            logger.info("[Watcher] Detected new PDF: %s", event.src_path)
            self.orchestrator.trigger_pdf_cooldown()

class DraftHandler(FileSystemEventHandler):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def on_created(self, event):
        self.handle_event(event)

    def on_modified(self, event):
        self.handle_event(event)

    def handle_event(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".txt"):
            if not event.src_path.lower().endswith("_cited.txt"):
                logger.info("[Watcher] Detected draft update: %s", event.src_path)
                self.orchestrator.trigger_draft_run(event.src_path)

class Orchestrator:
    def __init__(self):
        self.pdf_timer = None
        self.pdf_lock = threading.Lock()
        self.is_processing_pdf = False
        self.pdf_run_pending = False
        
        self.draft_timers = {}
        self.draft_lock = threading.Lock()

    def trigger_pdf_cooldown(self):
        with self.pdf_lock:
            if self.is_processing_pdf:
                logger.info("[Watcher] Pipeline currently running. File queued for NEXT run.")
                self.pdf_run_pending = True
            else:
                self.schedule_pdf_run()

    def schedule_pdf_run(self):
        if self.pdf_timer:
            self.pdf_timer.cancel()
        logger.info(
            "[Watcher] Pipeline scheduled to run in %ds… (Drop more files to reset timer)",
            PDF_COOLDOWN_SECONDS,
        )
        self.pdf_timer = threading.Timer(PDF_COOLDOWN_SECONDS, self.run_pipeline)
        self.pdf_timer.start()

    def run_pipeline(self):
        with self.pdf_lock:
            self.is_processing_pdf = True
            self.pdf_run_pending = False

        try:
            logger.info("Notifying Supervisor Agent of new PDFs…")
            process_event("New PDFs have been added to the raw/ directory. Please extract citations from them, fetch the papers, and ingest them.")
        except Exception as e:
            logger.error("UNEXPECTED ERROR: %s", e)
        finally:
            with self.pdf_lock:
                self.is_processing_pdf = False
                if self.pdf_run_pending:
                    logger.info("[Watcher] Resolving pending queued PDFs…")
                    self.schedule_pdf_run()

    def trigger_draft_run(self, filepath):
        with self.draft_lock:
            if filepath in self.draft_timers:
                self.draft_timers[filepath].cancel()
            
            # Start a short debounce timer
            timer = threading.Timer(DRAFT_COOLDOWN_SECONDS, self.run_agent5, args=[filepath])
            self.draft_timers[filepath] = timer
            timer.start()

    def run_agent5(self, filepath):
        try:
            logger.info("Notifying Supervisor Agent of new draft: %s…", filepath)
            process_event(f"A new draft text file needs citation processing. The file is located at: {filepath}. Please use the batch cite tool.")
        except Exception as e:
            logger.error("UNEXPECTED ERROR during draft citing: %s", e)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)
    
    orchestrator = Orchestrator()
    observer = Observer()
    
    # Schedule PDF watcher (raw/ — for automated pipeline)
    pdf_handler = PDFHandler(orchestrator)
    observer.schedule(pdf_handler, path=RAW_DIR, recursive=False)
    
    # Schedule Draft watcher
    draft_handler = DraftHandler(orchestrator)
    observer.schedule(draft_handler, path=DRAFTS_DIR, recursive=False)

    # Schedule Manual PDF watcher (pulled_pdfs/ — for user-placed PDFs)
    manual_handler = ManualPDFHandler()
    observer.schedule(manual_handler, path=PULLED_PDFS_DIR, recursive=False)
    
    logger.info("Starting Master Orchestrator (Tri-Mode)…")
    logger.info(" - Monitoring '%s/' for new source PDFs (Cooldown: %ds, Workers: %d)", RAW_DIR, PDF_COOLDOWN_SECONDS, DEFAULT_WORKERS)
    logger.info(" - Monitoring '%s/' for text drafts (Cooldown: %ds)", DRAFTS_DIR, DRAFT_COOLDOWN_SECONDS)
    logger.info(" - Monitoring '%s/' for manually placed PDFs (Agent 6)", PULLED_PDFS_DIR)
    logger.info("Press Ctrl+C to stop.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping Orchestrator…")
        if orchestrator.pdf_timer:
            orchestrator.pdf_timer.cancel()
        for timer in orchestrator.draft_timers.values():
            timer.cancel()
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
