import json
import os
import requests
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

from config import (
    EXTRACTED_CITATIONS_PATH,
    DOWNLOADED_JSON_PATH,
    FAILED_DOWNLOADS_PATH,
    PULLED_PDFS_DIR,
    UNPAYWALL_EMAIL,
    MAX_CITATION_LEN,
    ARXIV_RATE_LIMIT,
    UNPAYWALL_SLEEP,
)
from shared.log import get_logger

logger = get_logger("agent2")


def _checkpoint(downloaded, failed):
    """Write state to disk after every paper — crash-safe incremental saves."""
    with open(DOWNLOADED_JSON_PATH, "w") as f:
        json.dump(downloaded, f, indent=4)
    with open(FAILED_DOWNLOADS_PATH, "w") as f:
        json.dump(failed, f, indent=4)


def fetch_papers():
    if UNPAYWALL_EMAIL == "your-email@example.com":
        logger.warning(
            "UNPAYWALL_EMAIL is unset — Unpaywall rejects requests without a real "
            "contact address, so stage 2 will fail and every lookup will fall back "
            "to arXiv. Export UNPAYWALL_EMAIL=you@institution.edu to fix."
        )

    with open(EXTRACTED_CITATIONS_PATH, "r") as f:
        citations = json.load(f)
        
    citations = [c for c in citations if len(c) < MAX_CITATION_LEN]
    logger.info("Loaded %d valid citations to fetch.", len(citations))
    
    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)
    
    # ── FIX #1: Merge with existing state instead of overwriting ──────────
    if os.path.exists(DOWNLOADED_JSON_PATH):
        with open(DOWNLOADED_JSON_PATH, "r") as f:
            downloaded = json.load(f)
        logger.info("Resuming from existing state: %d papers already downloaded.", len(downloaded))
    else:
        downloaded = {}

    if os.path.exists(FAILED_DOWNLOADS_PATH):
        with open(FAILED_DOWNLOADS_PATH, "r") as f:
            failed = json.load(f)
    else:
        failed = []

    # Skip citations we've already successfully fetched
    already_fetched = set(downloaded.values())
    already_failed = {entry["citation"] for entry in failed}
    remaining = [c for c in citations if c not in already_fetched and c not in already_failed]

    if len(remaining) < len(citations):
        logger.info(
            "Skipping %d already-processed citations. %d remaining.",
            len(citations) - len(remaining),
            len(remaining),
        )

    for i, citation in enumerate(remaining):
        logger.info("[%d/%d] Processing: %s…", i + 1, len(remaining), citation[:60])
        pdf_saved = False
        fail_reason = ""
        
        try:
            # 1. Query Crossref to find DOI
            crossref_url = "https://api.crossref.org/works"
            params = {"query.bibliographic": citation, "rows": 1, "select": "DOI,title"}
            res = requests.get(crossref_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            doi = None
            title = "Unknown"
            
            if data["message"]["items"]:
                item = data["message"]["items"][0]
                doi = item.get("DOI")
                title = item.get("title", ["Unknown"])[0]
                
            if doi:
                # 2. Query Unpaywall using the DOI
                unpaywall_url = f"https://api.unpaywall.org/v2/{doi}"
                u_params = {"email": UNPAYWALL_EMAIL}
                u_res = requests.get(unpaywall_url, params=u_params, timeout=10)
                
                if u_res.status_code == 200:
                    u_data = u_res.json()
                    if u_data.get("is_oa") and u_data.get("best_oa_location"):
                        pdf_url = u_data["best_oa_location"].get("url_for_pdf")
                        if pdf_url:
                            # 3. Download the PDF
                            pdf_res = requests.get(pdf_url, stream=True, timeout=15)
                            if pdf_res.status_code == 200:
                                safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:50]
                                filename = os.path.join(PULLED_PDFS_DIR, f"{safe_title}_{i}.pdf")
                                with open(filename, "wb") as pdf_file:
                                    for chunk in pdf_res.iter_content(chunk_size=8192):
                                        pdf_file.write(chunk)
                                downloaded[filename] = citation
                                logger.info(" --> SUCCESS! Saved to %s", filename)
                                pdf_saved = True
                            else:
                                fail_reason = f"PDF download link failed (status {pdf_res.status_code})"
                        else:
                            fail_reason = "Open Access, but no direct PDF URL"
                    else:
                        fail_reason = "Paywalled / Not Open Access"
                else:
                    fail_reason = "Unpaywall lookup failed"
            else:
                fail_reason = "No DOI found in Crossref"

            # 4. Fallback to arXiv if Unpaywall failed or NO DOI
            if not pdf_saved:
                logger.info("     -> Unpaywall missed it (%s). Trying arXiv fallback…", fail_reason)
                query = f'ti:"{title}"' if title != "Unknown" else f'all:"{citation}"'
                safe_query = urllib.parse.quote(query)
                arxiv_url = f'http://export.arxiv.org/api/query?search_query={safe_query}&max_results=1'
                
                arxiv_res = requests.get(arxiv_url, timeout=10)
                if arxiv_res.status_code == 200:
                    root = ET.fromstring(arxiv_res.content)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    entries = root.findall('atom:entry', ns)
                    
                    if entries:
                        entry = entries[0]
                        pdf_link = None
                        for link in entry.findall('atom:link', ns):
                            if link.attrib.get('title') == 'pdf':
                                pdf_link = link.attrib.get('href')
                                break
                                
                        if pdf_link:
                            pdf_link = pdf_link.replace("http://", "https://")
                            if not pdf_link.endswith('.pdf'):
                                pdf_link += '.pdf'
                                
                            a_res = requests.get(pdf_link, stream=True, timeout=15)
                            if a_res.status_code == 200:
                                safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:50]
                                filename = os.path.join(PULLED_PDFS_DIR, f"{safe_title}_{i}_arxiv.pdf")
                                with open(filename, "wb") as pdf_file:
                                    for chunk in a_res.iter_content(chunk_size=8192):
                                        pdf_file.write(chunk)
                                        
                                downloaded[filename] = citation
                                logger.info(" --> SUCCESS via arXiv! Saved to %s", filename)
                                pdf_saved = True
                                time.sleep(ARXIV_RATE_LIMIT)
                            else:
                                fail_reason = f"{fail_reason} | arXiv PDF download failed"
                        else:
                            fail_reason = f"{fail_reason} | No PDF link on arXiv"
                    else:
                        fail_reason = f"{fail_reason} | Not found on arXiv"
                else:
                    fail_reason = f"{fail_reason} | arXiv API error"
            
            if not pdf_saved:
                failed.append({"citation": citation, "reason": fail_reason or "Fetch failed"})

        except Exception as e:
            failed.append({"citation": citation, "reason": str(e)})
            logger.error(" --> ERROR: %s", e)
            
        # ── FIX #7: Checkpoint after every paper ─────────────────────────
        _checkpoint(downloaded, failed)

        if pdf_saved:
            time.sleep(UNPAYWALL_SLEEP)
        else:
            time.sleep(ARXIV_RATE_LIMIT)

    logger.info("Done! Downloaded %d, Failed %d", len(downloaded), len(failed))

if __name__ == "__main__":
    fetch_papers()
