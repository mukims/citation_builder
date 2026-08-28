import os
import json
import argparse
import multiprocessing

from config import DOWNLOADED_JSON_PATH
from shared.log import get_logger
from shared.ingestion import ingest_pdfs

logger = get_logger("agent3")


def run_ingestor(workers=1, force=False):
    """Ingest every PDF Agent 2 successfully downloaded.

    Args:
        workers: Parallel worker processes for the parsing stage.
        force:   Re-process PDFs even if they are already recorded as ingested.
    """
    if not os.path.exists(DOWNLOADED_JSON_PATH):
        logger.info(
            "No download manifest at %s — run Agent 2 (python agent2_fetcher.py) "
            "first to fetch papers.", DOWNLOADED_JSON_PATH,
        )
        return

    with open(DOWNLOADED_JSON_PATH, "r") as f:
        downloaded = json.load(f)

    if not downloaded:
        logger.info("Download manifest is empty — nothing to ingest.")
        return

    if force:
        logger.info("--force: re-processing all PDFs regardless of ingestion status.")

    result = ingest_pdfs(downloaded, workers=workers, skip_ingested=not force)

    logger.info(
        "Done. Processed %d, skipped %d, inserted %d chunk(s), %d unreadable.",
        result["processed"], result["skipped"], result["inserted"], len(result["failed"]),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 3 — Ingestor with parallelization")
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
    run_ingestor(workers=args.workers, force=args.force)
