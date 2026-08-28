import subprocess
import re
import json
import os
import glob
import shutil

from config import RAW_DIR, EXTRACTED_CITATIONS_PATH
from shared.log import get_logger

logger = get_logger("agent1")

# Multiple reference-line patterns to support common citation formats
CITATION_PATTERNS = [
    re.compile(r'^\[(\d+)\]\s(.*)'),    # [1] Author...
    re.compile(r'^(\d+)\.\s{1,3}(.+)'), # 1. Author...
]


def _match_citation_line(line):
    """Try each citation pattern and return (id, rest_of_line) or None."""
    for pattern in CITATION_PATTERNS:
        m = pattern.match(line)
        if m:
            return int(m.group(1)), m.group(2)
    return None


def extract_citations(pdf_path):
    logger.info("Extracting text from %s using pdftotext…", pdf_path)
    result = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error("pdftotext failed: %s", result.stderr)
        return []

    lines = result.stdout.split('\n')
    
    citations = []
    current_citation = ""
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        match = _match_citation_line(line)
        if match:
            # We found a citation marker — save the previous one
            if current_citation:
                citations.append(current_citation.strip())
            
            current_citation = line
        elif current_citation:
            # Continue accumulating lines for the current citation
            # Stop if we hit something that clearly isn't part of a citation
            # e.g., a page number or random header, usually short.
            if len(line) < 3 and line.isdigit():
                continue  # Likely a page number
            
            current_citation += " " + line

    # Save the very last citation
    if current_citation:
        citations.append(current_citation.strip())
        
    logger.info("Extracted %d citation entries from %s.", len(citations), pdf_path)
    return citations

def _unique_destination(directory: str, filename: str) -> str:
    """Return a path in *directory* for *filename* that overwrites nothing.

    Two source papers can share a basename, and the previous `shutil.move`
    replaced the earlier file without a word.
    """
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{n}{ext}")
        n += 1
    return candidate


def run_extractor():
    all_citations = []
    
    # Load existing citations to append to
    if os.path.exists(EXTRACTED_CITATIONS_PATH):
        try:
            with open(EXTRACTED_CITATIONS_PATH, "r") as f:
                all_citations = json.load(f)
        except json.JSONDecodeError:
            pass

    processed_dir = os.path.join(RAW_DIR, "processed")
    failed_dir = os.path.join(RAW_DIR, "failed")
    os.makedirs(processed_dir, exist_ok=True)

    pdfs_processed = 0
    pdfs_failed = 0
    for raw_pdf in glob.glob(os.path.join(RAW_DIR, "*.pdf")):
        name = os.path.basename(raw_pdf)
        citations = extract_citations(raw_pdf)

        if citations:
            all_citations.extend(citations)
            # Move out of raw/ so the next run does not re-read it.
            shutil.move(raw_pdf, _unique_destination(processed_dir, name))
            pdfs_processed += 1
        else:
            # Nothing came out: pdftotext failed, the PDF is a scan with no text
            # layer, or its reference list is in a format the patterns miss.
            # Filing it under processed/ would claim a success it did not have
            # and leave the paper silently unaccounted for, so it goes somewhere
            # visible instead — still out of raw/, so watchers do not loop on it.
            os.makedirs(failed_dir, exist_ok=True)
            shutil.move(raw_pdf, _unique_destination(failed_dir, name))
            pdfs_failed += 1
            logger.warning(
                "No citations extracted from %s — moved to raw/failed/. "
                "Check that it has a text layer and a recognisable reference list.",
                name,
            )

    if pdfs_processed == 0:
        if pdfs_failed:
            logger.error(
                "%d PDF(s) yielded no citations; see raw/failed/. Nothing written.",
                pdfs_failed,
            )
        else:
            logger.info("No new PDFs found in %s.", RAW_DIR)
        return

    # Remove duplicates while maintaining some relative order
    unique_citations = list(dict.fromkeys(all_citations))
    
    with open(EXTRACTED_CITATIONS_PATH, "w") as f:
        json.dump(unique_citations, f, indent=4)
        
    logger.info(
        "Processed %d PDF(s)%s. Saved %d unique citations to %s",
        pdfs_processed,
        f" ({pdfs_failed} yielded nothing — see raw/failed/)" if pdfs_failed else "",
        len(unique_citations),
        EXTRACTED_CITATIONS_PATH,
    )
    return unique_citations

if __name__ == "__main__":
    run_extractor()
