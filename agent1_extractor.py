import os
import re
import json
import glob
from difflib import SequenceMatcher

from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup

from config import RAW_DIR, EXTRACTED_CITATIONS_PATH
from shared.log import get_logger

logger = get_logger("agent1")

# GROBID's TEI output. Kept alongside the PDFs so a run can be inspected
# and re-parsed without hitting the server again.
XML_OUTPUT_DIR = os.path.join(RAW_DIR, "grobid_output")

TEI_SUFFIXES = (
    ".references.tei.xml",
    ".fulltext.tei.xml",
    ".grobid.tei.xml",
    ".tei.xml",
)

# Reference types that rarely have a real DOI. A consolidated DOI on one of
# these is a likely false match rather than a find.
GREY_MARKERS = (
    "available online",
    "accessed on",
    "http://",
    "https://",
    "www.",
    "technical report",
    "white paper",
    "thesis",
    "dissertation",
    "standard",
    "patent",
    "datasheet",
)


def run_grobid_batch(pdf_dir: str, output_dir: str):
    """Send every PDF in pdf_dir to the local GROBID server."""
    logger.info("Sending PDFs in %s to GROBID…", pdf_dir)
    client = GrobidClient()

    # processFulltextDocument returns header, body and bibliography, and tags
    # in-text citation markers with the reference they point to.
    client.process(
        service="processFulltextDocument",
        input_path=pdf_dir,
        output=output_dir,
        n=10,
        consolidate_citations=1,
        consolidate_header=True,
        include_raw_citations=True,
        force=True,
    )
    logger.info("GROBID batch complete.")


def _clean(node):
    """Collapse GROBID's line wrapping into single-spaced text."""
    if node is None:
        return None
    text = " ".join(node.text.split())
    return text or None


def _stem(path):
    name = os.path.basename(path)
    for suffix in TEI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _person_name(pers):
    parts = [_clean(f) for f in pers.find_all("forename")]
    parts.append(_clean(pers.find("surname")))
    name = " ".join(p for p in parts if p)
    return name or None


def _normalise(text):
    """Lowercase alphanumeric tokens, for loose title comparison."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _doi_confidence(title, raw_reference):
    """
    Rate how well a consolidated title matches the reference as printed.

    GROBID's consolidation can match grey literature to an unrelated DOI and
    overwrite the title with the matched record. Comparing the returned title
    against the raw string catches most of those.

    Returns one of: "high", "medium", "low", "unknown".
    """
    if not raw_reference:
        return "unknown"

    raw_lower = raw_reference.lower()
    is_grey = any(marker in raw_lower for marker in GREY_MARKERS)

    if not title:
        return "low" if is_grey else "unknown"

    title_tokens = _normalise(title)
    raw_tokens = _normalise(raw_reference)
    if not title_tokens:
        return "unknown"

    overlap = len(title_tokens & raw_tokens) / len(title_tokens)
    ratio = SequenceMatcher(None, title.lower(), raw_lower).ratio()

    if overlap >= 0.8:
        return "medium" if is_grey else "high"
    if overlap >= 0.5 or ratio >= 0.4:
        return "medium"
    return "low"


def parse_reference(bibl, source_file=None):
    """Parse a single <biblStruct> from a TEI reference list."""
    analytic = bibl.find("analytic")
    monogr = bibl.find("monogr")

    # Article title lives in <analytic>; for books and reports the title is
    # in <monogr> instead, so fall back rather than returning None.
    title = None
    if analytic:
        title = _clean(analytic.find("title", type="main")) or _clean(analytic.find("title"))
    if not title and monogr:
        title = _clean(monogr.find("title", level="m")) or _clean(monogr.find("title"))

    container = None
    if monogr:
        journal = monogr.find("title", level="j")
        container = _clean(journal) if journal else None
        if container == title:
            container = None

    doi_node = bibl.find("idno", type="DOI")
    doi = doi_node.text.strip().lower() if doi_node and doi_node.text else None

    authors = []
    scope = analytic or bibl
    for author in scope.find_all("author"):
        pers = author.find("persName")
        if pers:
            name = _person_name(pers)
            if name:
                authors.append(name)

    year = None
    date = bibl.find("date", type="published")
    if date:
        when = date.get("when") or _clean(date) or ""
        match = re.search(r"(1[89]\d{2}|20\d{2})", when)
        if match:
            year = int(match.group(1))

    raw_node = bibl.find("note", type="raw_reference")
    raw_reference = _clean(raw_node)

    return {
        "source_file": source_file,
        "xml_id": bibl.get("{http://www.w3.org/XML/1998/namespace}id") or bibl.get("id"),
        "title": title,
        "container": container,
        "authors": authors,
        "year": year,
        "doi": doi,
        "doi_confidence": _doi_confidence(title, raw_reference) if doi else None,
        "raw_reference": raw_reference,
    }


def parse_article_metadata(soup, source_file=None):
    """Parse the citing paper's own title, authors and DOI from the TEI header."""
    header = soup.find("teiHeader")
    analytic = None
    if header:
        source_desc = header.find("sourceDesc")
        if source_desc:
            bibl = source_desc.find("biblStruct")
            if bibl:
                analytic = bibl.find("analytic")

    if analytic is None:
        return {"source_file": source_file, "title": None, "doi": None, "authors": []}

    doi_node = analytic.find("idno", type="DOI")
    authors = []
    for author in analytic.find_all("author"):
        pers = author.find("persName")
        if pers:
            name = _person_name(pers)
            if name:
                authors.append(name)

    return {
        "source_file": source_file,
        "title": _clean(analytic.find("title", type="main")) or _clean(analytic.find("title")),
        "doi": doi_node.text.strip().lower() if doi_node and doi_node.text else None,
        "authors": authors,
    }


def parse_tei_file(tei_file_path):
    """Parse one TEI file into its article metadata and its reference list."""
    with open(tei_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    source_file = f"{_stem(tei_file_path)}.pdf"
    article = parse_article_metadata(soup, source_file=source_file)

    list_bibl = soup.find("listBibl")
    references = []
    if list_bibl is not None:
        references = [
            parse_reference(bibl, source_file=source_file)
            for bibl in list_bibl.find_all("biblStruct")
        ]

    return article, references


def summarise(references):
    """Counts worth logging after a run."""
    total = len(references)
    with_doi = [r for r in references if r["doi"]]
    suspect = [r for r in with_doi if r["doi_confidence"] in ("low", "unknown")]
    return {
        "references": total,
        "with_doi": len(with_doi),
        "doi_coverage": round(len(with_doi) / total, 3) if total else 0.0,
        "suspect_doi": len(suspect),
        "suspect_ids": [(r["source_file"], r["xml_id"]) for r in suspect],
    }


def run_extractor():
    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

    pdfs = glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    if not pdfs:
        logger.info("No PDFs found in %s.", RAW_DIR)
        return

    run_grobid_batch(RAW_DIR, XML_OUTPUT_DIR)

    tei_files = sorted(glob.glob(os.path.join(XML_OUTPUT_DIR, "*.xml")))
    if not tei_files:
        logger.error(
            "GROBID produced no TEI in %s. Check the server is up: "
            "curl localhost:8070/api/isalive",
            XML_OUTPUT_DIR,
        )
        return

    articles = []
    references = []
    for tei_path in tei_files:
        try:
            article, refs = parse_tei_file(tei_path)
        except Exception:
            logger.exception("Failed to parse %s", tei_path)
            continue

        articles.append(article)
        references.extend(refs)

        if not refs:
            logger.warning("No references found in %s.", os.path.basename(tei_path))
        else:
            logger.info(
                "%s: %d references, %d with DOI.",
                os.path.basename(tei_path),
                len(refs),
                sum(1 for r in refs if r["doi"]),
            )

    payload = {"articles": articles, "references": references}
    with open(EXTRACTED_CITATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    stats = summarise(references)
    logger.info(
        "Processed %d PDF(s). %d references, %.1f%% with a DOI, %d suspect. Saved to %s",
        len(articles),
        stats["references"],
        stats["doi_coverage"] * 100,
        stats["suspect_doi"],
        EXTRACTED_CITATIONS_PATH,
    )
    if stats["suspect_ids"]:
        logger.warning("DOIs worth checking by hand: %s", stats["suspect_ids"][:20])


if __name__ == "__main__":
    run_extractor()