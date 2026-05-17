import os
import json
import argparse
import multiprocessing
import concurrent.futures
from tqdm import tqdm

from config import DOWNLOADED_JSON_PATH, DETECTRON_WEIGHTS, IMAGES_DIR
from shared.log import get_logger
from shared.ingestion import process_pdf, upsert_corpus, rebuild_bm25

logger = get_logger("agent3")


def run_ingestor(workers=1):
    with open(DOWNLOADED_JSON_PATH, "r") as f:
        downloaded = json.load(f)
    
    corpus = []
    
    if workers <= 1:
        logger.info("Running sequentially (1 worker)…")
        for pdf_path, citation_string in downloaded.items():
            if not os.path.exists(pdf_path):
                continue
            corpus.extend(process_pdf(pdf_path, citation_string))
    else:
        logger.info("Running in parallel with %d workers…", workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for pdf_path, citation_string in downloaded.items():
                if not os.path.exists(pdf_path):
                    continue
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
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    run_ingestor(workers=args.workers)
