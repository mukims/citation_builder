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

from agent_graph import process_event

COOLDOWN_SECONDS = 30
DRAFT_COOLDOWN_SECONDS = 2
WORKERS = 4
RAW_DIR = "raw"
DRAFTS_DIR = "drafts"
WORKERS = 4
RAW_DIR = "raw"
DRAFTS_DIR = "drafts"

class PDFHandler(FileSystemEventHandler):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            print(f"[Watcher] Detected new PDF: {event.src_path}")
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
                print(f"[Watcher] Detected draft update: {event.src_path}")
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
                print("[Watcher] Pipeline currently running. File queued for NEXT run.")
                self.pdf_run_pending = True
            else:
                self.schedule_pdf_run()

    def schedule_pdf_run(self):
        if self.pdf_timer:
            self.pdf_timer.cancel()
        print(f"[Watcher] Pipeline scheduled to run in {COOLDOWN_SECONDS} seconds... (Drop more files to reset timer)")
        self.pdf_timer = threading.Timer(COOLDOWN_SECONDS, self.run_pipeline)
        self.pdf_timer.start()

    def run_pipeline(self):
        with self.pdf_lock:
            self.is_processing_pdf = True
            self.pdf_run_pending = False

        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Notifying Supervisor Agent of new PDFs...")
            process_event("New PDFs have been added to the raw/ directory. Please extract citations from them, fetch the papers, and ingest them.")
        except Exception as e:
            print(f"\n[Orchestrator] UNEXPECTED ERROR: {e}")
        finally:
            with self.pdf_lock:
                self.is_processing_pdf = False
                if self.pdf_run_pending:
                    print("[Watcher] Resolving pending queued PDFs...")
                    self.schedule_pdf_run()

    def trigger_draft_run(self, filepath):
        with self.draft_lock:
            if filepath in self.draft_timers:
                self.draft_timers[filepath].cancel()
            
            # Start a short 2-second debounce timer
            timer = threading.Timer(DRAFT_COOLDOWN_SECONDS, self.run_agent5, args=[filepath])
            self.draft_timers[filepath] = timer
            timer.start()

    def run_agent5(self, filepath):
        try:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Notifying Supervisor Agent of new draft: {filepath}...")
            process_event(f"A new draft text file needs citation processing. The file is located at: {filepath}. Please use the batch cite tool.")
        except Exception as e:
            print(f"\n[Orchestrator] UNEXPECTED ERROR during draft citing: {e}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    
    orchestrator = Orchestrator()
    observer = Observer()
    
    # Schedule PDF watcher
    pdf_handler = PDFHandler(orchestrator)
    observer.schedule(pdf_handler, path=RAW_DIR, recursive=False)
    
    # Schedule Draft watcher
    draft_handler = DraftHandler(orchestrator)
    observer.schedule(draft_handler, path=DRAFTS_DIR, recursive=False)
    
    print(f"Starting Master Orchestrator (Dual Mode)...")
    print(f" - Monitoring '{RAW_DIR}/' for new PDFs (Cooldown: {COOLDOWN_SECONDS}s, Workers: {WORKERS})")
    print(f" - Monitoring '{DRAFTS_DIR}/' for text drafts (Cooldown: {DRAFT_COOLDOWN_SECONDS}s)")
    print("Press Ctrl+C to stop.\n")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Orchestrator...")
        if orchestrator.pdf_timer:
            orchestrator.pdf_timer.cancel()
        for timer in orchestrator.draft_timers.values():
            timer.cancel()
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
