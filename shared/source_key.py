import hashlib
import re
import unicodedata

# Words that vary between how a reference is printed and how a paper's own
# header states its title. Stripped before hashing so the two still agree.
_TITLE_NOISE = re.compile(
    r"\b(a|an|the|of|for|and|on|in|to|with|via|using|towards|toward)\b"
)


def _fold(text):
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _fold_title(title):
    folded = _fold(title)
    folded = _TITLE_NOISE.sub(" ", folded)
    return " ".join(folded.split())


def _hash(prefix, payload, length=16):
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def normalise_doi(doi):
    """Strip the common prefixes so the same DOI always folds to one string."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "doi "):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    doi = doi.strip().rstrip(".")
    return doi if doi.startswith("10.") else None


def source_key(record, min_doi_confidence=("high", "medium")):
    """
    Return a deterministic key for one reference or ingested document.

    The prefix says where the key came from, so a consumer can tell an
    authoritative identifier from a derived one:

        doi:10.1039/d4nr01234a      resolved, trustworthy
        arxiv:2401.12345            resolved, trustworthy
        url:<hash>                  a specific retrievable resource
        title:<hash>                derived from title + first author + year
        raw:<hash>                  nothing parsed; last resort

    Same input always gives the same key, so runs are comparable.
    """
    doi = normalise_doi(record.get("doi"))
    confidence = record.get("doi_confidence")
    if doi and (confidence is None or confidence in min_doi_confidence):
        return f"doi:{doi}"

    arxiv = record.get("arxiv_id")
    if arxiv:
        return f"arxiv:{arxiv.strip().lower().replace('arxiv:', '')}"

    pmid = record.get("pmid")
    if pmid:
        return f"pmid:{str(pmid).strip()}"

    url = record.get("url")
    if url:
        cleaned = url.strip().lower().split("#")[0].rstrip("/")
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        return _hash("url", cleaned)

    title = _fold_title(record.get("title"))
    if title:
        authors = record.get("authors") or []
        # Surname of the first author only — initials and middle names are
        # the parts most likely to differ between two parses of one paper.
        first_author = _fold(authors[0].split()[-1]) if authors else ""
        year = str(record.get("year") or "")
        return _hash("title", f"{title}|{first_author}|{year}")

    raw = _fold(record.get("raw_reference"))
    if raw:
        return _hash("raw", raw)

    return None


def is_authoritative(key):
    """True when the key came from a real identifier rather than a derived one."""
    return bool(key) and key.split(":", 1)[0] in ("doi", "arxiv", "pmid")


def title_blocking_key(record):
    """
    A coarse key for finding near-duplicates that hashing alone will miss.

    Two parses of one paper can differ by a subtitle or a dropped word, which
    changes the title hash. Group candidates on this, then compare properly
    within each group.
    """
    title = _fold_title(record.get("title"))
    if not title:
        return None
    tokens = [t for t in title.split() if len(t) > 3][:4]
    return " ".join(sorted(tokens)) or None