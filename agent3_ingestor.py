import os
import json
import argparse
import multiprocessing
import concurrent.futures
from tqdm import tqdm

from config import DOWNLOADED_JSON_PATH
from shared.log import get_logger
from shared.ingestion import process_pdf, upsert_corpus, rebuild_bm25, get_ingested_documents

logger = get_logger("agent3")


def run_ingestor(workers=1):
    with open(DOWNLOADED_JSON_PATH, "r") as f:
        downloaded = json.load(f)

    # Skip PDFs that are already fully ingested in ChromaDB
    already_ingested = get_ingested_documents()
    to_process = {}
    skipped = 0
    for pdf_path, citation_string in downloaded.items():
        if not os.path.exists(pdf_path):
            continue
        pdf_name = os.path.basename(pdf_path).strip().replace(" ", "_").lower()
        if pdf_name in already_ingested:
            skipped += 1
            continue
        to_process[pdf_path] = citation_string

    if skipped:
        logger.info(
            "Skipping %d already-ingested PDFs. %d remaining to process.",
            skipped, len(to_process),
        )

    if not to_process:
        logger.info("All PDFs already ingested. Nothing to do.")
        return

    corpus = []

    if workers <= 1:
        logger.info("Running sequentially (1 worker) on %d PDFs…", len(to_process))
        for pdf_path, citation_string in to_process.items():
            corpus.extend(process_pdf(pdf_path, citation_string))
    else:
        logger.info("Running in parallel with %d workers on %d PDFs…", workers, len(to_process))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for pdf_path, citation_string in to_process.items():
                futures.append(
                    executor.submit(process_pdf, pdf_path, citation_string)
                )

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing PDFs",
            ):
                try:
                    corpus.extend(future.result())
                except Exception as e:
                    logger.error("Worker failed: %s", e)

    # Chunk, embed, and upsert into ChromaDB
    inserted = upsert_corpus(corpus)
    logger.info("Inserted %d new chunks.", inserted)

    # Rebuild BM25 from the full collection
    rebuild_bm25()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestor with Parallelization")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Use 1 to disable parallelization, 2+ for multiprocessing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all PDFs, even if already ingested.",
    )
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    if args.force:
        logger.info("--force flag: re-processing all PDFs regardless of ingestion status.")
        # Temporarily monkey-patch to return empty set
        import shared.ingestion as _ing
        _ing.get_ingested_documents = lambda *a, **kw: set()
    run_ingestor(workers=args.workers)
