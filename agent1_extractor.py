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
    os.makedirs(processed_dir, exist_ok=True)
    
    pdfs_processed = 0
    for raw_pdf in glob.glob(os.path.join(RAW_DIR, "*.pdf")):
        citations = extract_citations(raw_pdf)
        all_citations.extend(citations)
        
        # Move to processed folder so we don't re-process it next time
        dest = os.path.join(processed_dir, os.path.basename(raw_pdf))
        shutil.move(raw_pdf, dest)
        pdfs_processed += 1
    
    if pdfs_processed == 0:
        logger.info("No new PDFs found in %s.", RAW_DIR)
        return

    # Remove duplicates while maintaining some relative order
    unique_citations = list(dict.fromkeys(all_citations))
    
    with open(EXTRACTED_CITATIONS_PATH, "w") as f:
        json.dump(unique_citations, f, indent=4)
        
    logger.info("Moved %d PDFs to processed/. Saved %d unique citations to %s", pdfs_processed, len(unique_citations), EXTRACTED_CITATIONS_PATH)

if __name__ == "__main__":
    run_extractor()
