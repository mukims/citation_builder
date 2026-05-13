import subprocess
import re
import json
import os
import glob

def extract_citations(pdf_path):
    print(f"Extracting text from {pdf_path} using pdftotext...")
    result = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running pdftotext: {result.stderr}")
        return []

    lines = result.stdout.split('\n')
    
    citations = []
    current_citation = ""
    in_references_section = False
    
    # We use a pattern to detect citation markers like "[1]" at the start of a line.
    citation_pattern = re.compile(r'^\[(\d+)\]\s(.*)')
    
    # Track the last ID seen to ensure we're getting contiguous lists 
    # (e.g. 1 to 35, then 1 to 23).
    last_id = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if not line:
            continue
            
        match = citation_pattern.match(line)
        if match:
            # We found a citation marker
            cid = int(match.group(1))
            
            # If we were tracking a citation, save it
            if current_citation:
                citations.append(current_citation.strip())
            
            current_citation = line
            last_id = cid
        elif current_citation:
            # Continue accumulating lines for the current citation
            # Stop if we hit something that clearly isn't part of a citation
            # e.g., a page number or random header, usually short.
            if len(line) < 3 and line.isdigit():
                continue # Likely a page number
            
            # If the line starts with a new abstract-like phrase or header, we might be out.
            # But usually references are contiguous blocks until the end of the file.
            current_citation += " " + line

    # Save the very last citation
    if current_citation:
        citations.append(current_citation.strip())
        
    print(f"Successfully extracted {len(citations)} citation entries.")
    return citations

def run_extractor():
    all_citations = []
    
    for raw_pdf in glob.glob("raw/*.pdf"):
        citations = extract_citations(raw_pdf)
        all_citations.extend(citations)
    
    # Remove duplicates while maintaining some relative order
    unique_citations = list(dict.fromkeys(all_citations))
    
    out_file = "extracted_citations.json"
    with open(out_file, "w") as f:
        json.dump(unique_citations, f, indent=4)
        
    print(f"Saved {len(unique_citations)} unique citations to {out_file}")

if __name__ == "__main__":
    run_extractor()
