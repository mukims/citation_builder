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
from shared.source_key import source_key, normalise_doi, is_authoritative

logger = get_logger("agent2")

# Crossref and arXiv both ask for identifying contact details. Being polite
# gets you the faster rate-limit pool rather than the shared one.
HEADERS = {"User-Agent": f"citation_builder/0.1 (mailto:{UNPAYWALL_EMAIL})"}
 
TRUSTED_CONFIDENCE = ("high", "medium")
 
 
def _checkpoint(downloaded, failed):
    """Write state to disk after every paper — crash-safe incremental saves."""
    with open(DOWNLOADED_JSON_PATH, "w") as f:
        json.dump(downloaded, f, indent=2, ensure_ascii=False)
    with open(FAILED_DOWNLOADS_PATH, "w") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
 
 
def _load_state(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("%s is unreadable; starting fresh.", path)
        return default
 
 
def _load_references():
    """Read agent1's output, tolerating the old flat-string format."""
    with open(EXTRACTED_CITATIONS_PATH, "r") as f:
        payload = json.load(f)
 
    if isinstance(payload, dict):
        return payload.get("references", [])
 
    # Legacy format: a list of raw citation strings.
    logger.warning("Legacy citation format detected; only raw strings available.")
    return [{"raw_reference": c, "title": None, "doi": None, "authors": []} for c in payload]
 
 
def _filename_for(key):
    """Stable, filesystem-safe filename derived from the source key."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", key)[:120] + ".pdf"
 
 
def _is_pdf(first_bytes):
    """Servers frequently return an HTML error page with status 200."""
    return first_bytes[:5] == b"%PDF-"
 
 
def _download_pdf(url, dest_path):
    """Stream a PDF to disk, refusing anything that isn't actually a PDF."""
    with requests.get(url, headers=HEADERS, stream=True, timeout=30) as res:
        if res.status_code != 200:
            return False, f"HTTP {res.status_code}"
 
        chunks = res.iter_content(chunk_size=8192)
        try:
            first = next(chunks)
        except StopIteration:
            return False, "empty response"
 
        if not _is_pdf(first):
            ctype = res.headers.get("Content-Type", "unknown")
            return False, f"not a PDF (content-type {ctype})"
 
        tmp_path = dest_path + ".part"
        with open(tmp_path, "wb") as fh:
            fh.write(first)
            for chunk in chunks:
                fh.write(chunk)
 
    os.replace(tmp_path, dest_path)
    return True, None
 
 
def resolve_doi(ref):
    """
    Return (doi, how). Trust agent1's DOI when it was confidently matched;
    otherwise ask Crossref using the parsed title, which is a better query
    than the whole raw reference string.
    """
    doi = normalise_doi(ref.get("doi"))
    if doi and ref.get("doi_confidence") in TRUSTED_CONFIDENCE:
        return doi, "grobid"
 
    query = ref.get("title") or ref.get("raw_reference")
    if not query:
        return None, None
 
    params = {"query.bibliographic": query, "rows": 1, "select": "DOI,title"}
    if ref.get("authors"):
        params["query.author"] = ref["authors"][0]
 
    try:
        res = requests.get(
            "https://api.crossref.org/works", params=params, headers=HEADERS, timeout=15
        )
        res.raise_for_status()
        items = res.json()["message"]["items"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("Crossref lookup failed: %s", exc)
        return doi, "grobid-unverified" if doi else (None, None)
 
    if not items:
        return (doi, "grobid-unverified") if doi else (None, None)
 
    return normalise_doi(items[0].get("DOI")), "crossref"
 


def try_unpaywall(doi, dest_path):
    """
    Look for an open-access PDF for this DOI.

    Unpaywall's best_oa_location is usually the publisher's own site, which
    is also the one most likely to block an automated request. Every known
    location is tried, repositories first.
    """
    try:
        res = requests.get(
            f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}",
            params={"email": UNPAYWALL_EMAIL},
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as exc:
        return False, f"Unpaywall error: {exc}"

    if res.status_code != 200:
        return False, f"Unpaywall lookup failed (HTTP {res.status_code})"

    try:
        data = res.json()
    except ValueError:
        return False, "Unpaywall returned malformed JSON"

    if not data.get("is_oa"):
        return False, "paywalled / not open access"

    # Collect every location with a PDF, best_oa_location included, then
    # de-duplicate on URL since it usually repeats one of the others.
    locations = list(data.get("oa_locations") or [])
    best = data.get("best_oa_location")
    if best:
        locations.append(best)

    candidates = []
    seen = set()
    for loc in locations:
        url = (loc or {}).get("url_for_pdf")
        if url and url not in seen:
            seen.add(url)
            candidates.append({"url": url, "host_type": loc.get("host_type") or "?"})

    if not candidates:
        return False, "open access, but no direct PDF URL"

    candidates.sort(key=lambda c: c["host_type"] != "repository")

    reasons = []
    for candidate in candidates:
        ok, reason = _download_pdf(candidate["url"], dest_path)
        if ok:
            return True, None
        reasons.append(f"{candidate['host_type']}: {reason}")

    return False, "Unpaywall PDF failed: " + "; ".join(reasons)



def try_europepmc(doi, ref, dest_path):
    """Europe PMC covers biomedical literature that arXiv does not."""
    query = f'DOI:"{doi}"' if doi else f'TITLE:"{ref.get("title", "")}"'
    try:
        res = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": 1,
                    "resultType": "core"},
            headers=HEADERS, timeout=20,
        )
        res.raise_for_status()
        results = res.json().get("resultList", {}).get("result", [])
    except (requests.RequestException, ValueError) as exc:
        return False, f"Europe PMC error: {exc}"

    if not results:
        return False, "not in Europe PMC"

    hit = results[0]
    pmcid = hit.get("pmcid")
    if not pmcid or hit.get("isOpenAccess") != "Y":
        return False, "in Europe PMC but not open access"

    url = (f"https://www.ebi.ac.uk/europepmc/webservices/rest/"
           f"{pmcid}/fullTextPdf")
    ok, reason = _download_pdf(url, dest_path)
    return ok, None if ok else f"Europe PMC PDF failed: {reason}"




def try_arxiv(ref, dest_path):
    """Fall back to arXiv, searching on title where we have one."""
    title = ref.get("title")
    raw = ref.get("raw_reference")
    if not (title or raw):
        return False, "nothing to search arXiv with"
 
    query = f'ti:"{title}"' if title else f'all:"{raw[:200]}"'
    url = (
        "http://export.arxiv.org/api/query?search_query="
        f"{urllib.parse.quote(query)}&max_results=1"
    )
 
    try:
        res = requests.get(url, headers=HEADERS, timeout=20)
        res.raise_for_status()
        root = ET.fromstring(res.content)
    except (requests.RequestException, ET.ParseError) as exc:
        return False, f"arXiv error: {exc}"
 
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        return False, "not found on arXiv"
 
    # arXiv title search is loose — confirm the result is actually the paper
    # we asked for before downloading it.
    if title:
        found = " ".join((entries[0].findtext("atom:title", "", ns)).split()).lower()
        wanted = " ".join(title.split()).lower()
        overlap = set(re.findall(r"[a-z0-9]+", found)) & set(re.findall(r"[a-z0-9]+", wanted))
        if len(overlap) < 0.6 * len(set(re.findall(r"[a-z0-9]+", wanted))):
            return False, "arXiv match was a different paper"
 
    pdf_link = None
    for link in entries[0].findall("atom:link", ns):
        if link.attrib.get("title") == "pdf":
            pdf_link = link.attrib.get("href", "").replace("http://", "https://")
            break
 
    if not pdf_link:
        return False, "no PDF link on arXiv"
    if not pdf_link.endswith(".pdf"):
        pdf_link += ".pdf"
 
    ok, reason = _download_pdf(pdf_link, dest_path)
    return ok, None if ok else f"arXiv PDF failed: {reason}"
 
 
def fetch_papers():
    references = _load_references()
    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)
 
    # One entry per distinct source, even when several papers cite it.
    unique = {}
    for ref in references:
        raw = ref.get("raw_reference") or ""
        if raw and len(raw) > MAX_CITATION_LEN:
            continue
        key = source_key(ref)
        if not key:
            continue
        unique.setdefault(key, ref)
 
    logger.info("%d references collapse to %d distinct sources.", len(references), len(unique))
 
    downloaded = _load_state(DOWNLOADED_JSON_PATH, {})
    failed = _load_state(FAILED_DOWNLOADS_PATH, {})
    if isinstance(failed, list):  # legacy shape
        failed = {}
 
    remaining = [(k, r) for k, r in unique.items() if k not in downloaded and k not in failed]
    if len(remaining) < len(unique):
        logger.info("Skipping %d already processed.", len(unique) - len(remaining))
 
    for i, (key, ref) in enumerate(remaining, start=1):
        label = (ref.get("title") or ref.get("raw_reference") or key)[:70]
        logger.info("[%d/%d] %s", i, len(remaining), label)
 
        dest = os.path.join(PULLED_PDFS_DIR, _filename_for(key))
        reasons = []
        saved = False
        doi = how = None
 
        try:
            doi, how = resolve_doi(ref)
 
            if doi:
                saved, reason = try_unpaywall(doi, dest)
                if not saved:
                    reasons.append(reason)
                    time.sleep(UNPAYWALL_SLEEP)
            else:
                reasons.append("no DOI resolved")
 
            if not saved:
                saved, reason = try_arxiv(ref, dest)
                if not saved:
                    reasons.append(reason)
                time.sleep(ARXIV_RATE_LIMIT)
 
        except Exception as exc:
            logger.exception("Unhandled error on %s", key)
            reasons.append(str(exc))
 
        record = {
            "key": key,
            "doi": doi,
            "doi_source": how,
            "authoritative": is_authoritative(key),
            "title": ref.get("title"),
            "cited_by": ref.get("source_file"),
            "xml_id": ref.get("xml_id"),
            "raw_reference": ref.get("raw_reference"),
        }
 
        if saved:
            record["path"] = dest
            downloaded[key] = record
            logger.info("    saved -> %s", os.path.basename(dest))
        else:
            record["reason"] = " | ".join(r for r in reasons if r) or "fetch failed"
            failed[key] = record
            logger.info("    unavailable: %s", record["reason"])
 
        _checkpoint(downloaded, failed)
 
    manual = [r for r in failed.values() if r.get("doi")]
    logger.info(
        "Done. %d downloaded, %d unavailable (%d have a DOI and can be fetched by hand).",
        len(downloaded),
        len(failed),
        len(manual),
    )
 
 
if __name__ == "__main__":
    fetch_papers()