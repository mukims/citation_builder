import os
import re
from difflib import SequenceMatcher

from bs4 import BeautifulSoup

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

    # How much of the consolidated title actually appears in the printed
    # reference. Token overlap is more forgiving of reordering than a
    # straight string ratio, so use it as the primary signal.
    overlap = len(title_tokens & raw_tokens) / len(title_tokens)
    ratio = SequenceMatcher(None, (title or "").lower(), raw_lower).ratio()

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

    record = {
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
    return record


def parse_reference_list(tei_file_path):
    """Parse every reference in a GROBID processReferences TEI file."""
    with open(tei_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    source_file = f"{_stem(tei_file_path)}.pdf"

    list_bibl = soup.find("listBibl")
    if list_bibl is None:
        return []

    return [
        parse_reference(bibl, source_file=source_file)
        for bibl in list_bibl.find_all("biblStruct")
    ]


def summarise(records):
    """Counts worth logging after a run, and worth quoting on Friday."""
    total = len(records)
    with_doi = [r for r in records if r["doi"]]
    by_conf = {}
    for r in with_doi:
        by_conf[r["doi_confidence"]] = by_conf.get(r["doi_confidence"], 0) + 1
    return {
        "references": total,
        "with_doi": len(with_doi),
        "doi_coverage": round(len(with_doi) / total, 3) if total else 0.0,
        "confidence": by_conf,
        "needs_review": [
            r["xml_id"] for r in with_doi if r["doi_confidence"] in ("low", "unknown")
        ],
    }


if __name__ == "__main__":
    import json
    import sys

    for path in sys.argv[1:]:
        records = parse_reference_list(path)
        print(json.dumps(summarise(records), indent=2))
        for r in records:
            print(json.dumps(r, ensure_ascii=False))