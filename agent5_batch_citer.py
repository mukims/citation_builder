import argparse
import json
import os
import re
from datetime import datetime

import ollama

from config import LLM_MODEL
from prompts import (
    CITATION_NEED_CHECK,
    CITE_SENTENCE_SYSTEM,
    CITE_SENTENCE_USER,
)
from shared.log import get_logger
from shared.db import load_search_resources
from shared.search import hybrid_search
from shared.retry import retry

logger = get_logger("agent5")


# ─── Improved sentence splitter ──────────────────────────────────────────────

# Abbreviations that should NOT trigger a sentence break
_ABBREVS = r"(?:et al|Fig|Figs|Eq|Eqs|Dr|Prof|Mr|Mrs|Ms|Jr|Sr|vs|i\.e|e\.g|cf|approx|Ref|Refs|Vol|No|Ch|Sec|pp)"

def split_into_sentences(text):
    """
    Split text into sentences, handling common scientific abbreviations
    that contain periods (e.g., "et al.", "Fig.", "Eq.").
    """
    _TOKEN = "<PD>"
    def replace_abbrev_period(match):
        return match.group(0).replace('.', _TOKEN)
    
    # Mask periods in known abbreviations
    pattern = rf'\b({_ABBREVS})\.'
    masked_text = re.sub(pattern, replace_abbrev_period, text)
    
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', masked_text.strip())
    return [s.replace(_TOKEN, '.').strip() for s in sentences if s.strip()]


# ─── Citation key extraction ─────────────────────────────────────────────────

_CITE_RE = re.compile(r"\\cite\{([^}]*)\}")


def _cite_keys(text: str) -> set[str]:
    """Return the set of citation keys actually present in *text*.

    Handles the multi-key form ``\\cite{a,b}`` as well as ``\\cite{a}``.
    An empty set means no citation was made, whatever the model replied.
    """
    keys = set()
    for match in _CITE_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


# ─── Batched citation-need check ─────────────────────────────────────────────

@retry(max_retries=2, backoff=2.0)
def _batch_needs_citation(sentences):
    """
    Ask the LLM once whether each sentence in a batch needs a citation.
    Returns a list of booleans aligned with the input list.
    """
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = CITATION_NEED_CHECK.format(numbered=numbered)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        answer = response.message.content
    except AttributeError:
        answer = response["message"]["content"]

    # Parse the YES/NO list.
    #
    # Index by the number the model emitted rather than by line position: a
    # preamble line, a blank line, or a double-spaced list would otherwise
    # shift every verdict onto the wrong sentence. Padding a short result to
    # length hides exactly that failure, so a missing verdict raises instead
    # (the @retry above gives the model two more attempts first).
    verdicts = {}
    for line in answer.strip().split("\n"):
        match = re.match(r"\s*(\d+)\s*[.)]\s*(YES|NO)\b", line.strip(), re.IGNORECASE)
        if not match:
            continue
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(sentences):
            verdicts[idx] = match.group(2).upper() == "YES"

    missing = [i + 1 for i in range(len(sentences)) if i not in verdicts]
    if missing:
        raise ValueError(
            f"Citation-need check returned no verdict for sentence(s) {missing} "
            f"of {len(sentences)}. Raw response:\n{answer.strip()[:500]}"
        )

    return [verdicts[i] for i in range(len(sentences))]


@retry(max_retries=3, backoff=2.0)
def _cite_sentence_with_reasoning(sentence, context_str):
    """Ask the LLM to rewrite a sentence with \\cite{key} and explain why.

    Returns:
        tuple: (cited_sentence, reasoning)
    """
    sys_prompt = CITE_SENTENCE_SYSTEM
    user_prompt = CITE_SENTENCE_USER.format(sentence=sentence, context=context_str)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    try:
        raw = response.message.content.strip()
    except AttributeError:
        raw = response["message"]["content"].strip()

    prompt_tokens = getattr(response, "prompt_eval_count", "N/A")
    completion_tokens = getattr(response, "eval_count", "N/A")
    logger.info(
        " -> [Token Stats] Submitted: %s | Generated: %s", prompt_tokens, completion_tokens
    )

    # Parse the structured response
    cited_sentence = sentence  # fallback
    reasoning = ""

    cited_match = re.search(r'CITED:\s*(.+?)(?:\nREASON:|$)', raw, re.DOTALL)
    reason_match = re.search(r'REASON:\s*(.+)', raw, re.DOTALL)

    if cited_match:
        cited_sentence = cited_match.group(1).strip()
    elif "\\cite" in raw:
        # If the model didn't follow format but did produce a citation,
        # use the first line as the sentence
        cited_sentence = raw.split("\n")[0].strip()

    if reason_match:
        reasoning = reason_match.group(1).strip()
    elif not cited_match and len(raw.split("\n")) > 1:
        # Try to extract reasoning from non-formatted response
        reasoning = " ".join(raw.split("\n")[1:]).strip()

    return cited_sentence, reasoning


def _generate_report(
    file_path: str,
    report_path: str,
    sentences: list[str],
    cited_sentences: list[str],
    needs_cite: list[bool],
    citation_entries: list[dict],
    citation_mapping: dict,
    key_registry: dict,
):
    """Generate a markdown report explaining every citation decision."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    basename = os.path.basename(file_path)

    # These three are mutually exclusive and cover every sentence, so a reader
    # can see at a glance how often the pipeline wanted a citation and failed
    # to produce one — previously indistinguishable from "none needed".
    n_cited = sum(1 for e in citation_entries if e.get("cited"))
    n_not_needed = sum(
        1 for i, e in enumerate(citation_entries)
        if not e.get("cited") and not needs_cite[i]
    )
    n_declined = sum(
        1 for i, e in enumerate(citation_entries)
        if not e.get("cited") and needs_cite[i]
    )

    lines = [
        f"# Citation Report — `{basename}`",
        f"*Generated on {timestamp}*\n",
        "---\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total sentences | {n_cited + n_not_needed + n_declined} |",
        f"| Cited | {n_cited} |",
        f"| No citation needed | {n_not_needed} |",
        f"| **Needed a citation, none made** | **{n_declined}** |",
        f"| Unique sources cited | {len(citation_mapping)} |",
        f"| Sources retrieved but never cited | {len(key_registry) - len(citation_mapping)} |",
        "",
        "## Citation Key Mapping\n",
        "| Key | Full Citation |",
        "|-----|---------------|",
    ]

    for full_cit, key in sorted(citation_mapping.items(), key=lambda x: x[1]):
        # Truncate long citations for the table
        short = full_cit[:100] + ("…" if len(full_cit) > 100 else "")
        lines.append(f"| `{key}` | {short} |")

    lines.append("")
    lines.append("---\n")
    lines.append("## Sentence-by-Sentence Analysis\n")

    for i, entry in enumerate(citation_entries):
        lines.append(f"### Sentence {i+1}\n")

        if not entry.get("cited"):
            lines.append(f"> {entry['original']}\n")
            lines.append(f"**Decision:** Not cited — {entry.get('skip_reason', 'skipped')}\n")

            if entry.get("reasoning"):
                lines.append("**Model's explanation:**")
                lines.append(f"{entry['reasoning']}\n")

            if entry.get("candidates"):
                lines.append("**Retrieved but not used:**")
                for src in entry["candidates"]:
                    lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                lines.append("")
        else:
            lines.append("**Original:**")
            lines.append(f"> {entry['original']}\n")
            lines.append("**With citation:**")
            lines.append(f"> {entry['cited_text']}\n")

            if entry.get("reasoning"):
                lines.append("**Reasoning:**")
                lines.append(f"{entry['reasoning']}\n")

            # Separate what was actually cited from what was merely retrieved —
            # listing all three candidates as "sources used" overstates the
            # evidence behind the sentence.
            if entry.get("sources"):
                used = _cite_keys(entry["cited_text"])
                chosen = [s for s in entry["sources"] if s["key"] in used]
                rejected = [s for s in entry["sources"] if s["key"] not in used]

                if chosen:
                    lines.append("**Cited:**")
                    for src in chosen:
                        lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                    lines.append("")
                if rejected:
                    lines.append("**Also retrieved, not cited:**")
                    for src in rejected:
                        lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                    lines.append("")

        lines.append("---\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Saved citation report to %s", report_path)


def run_batch_citer(file_path, out_path="cited_draft.txt"):
    """Cite every claim in *file_path* that the retrieved corpus supports.

    Writes three files alongside *out_path*: the cited draft, a
    ``_citations.json`` key mapping for BibTeX, and a ``_report.md`` explaining
    each decision.

    Returns:
        str | None: *out_path* if the draft was written, None if the run was
        aborted before producing output.
    """
    if not os.path.exists(file_path):
        logger.error("File %s not found.", file_path)
        return None

    with open(file_path, "r") as f:
        draft_text = f.read()

    collection, bm25, texts, metadatas = load_search_resources()

    sentences = split_into_sentences(draft_text)
    logger.info("Split draft into %d sentences.", len(sentences))

    # ── Batch citation-need check ────────────────────────────────────────
    eligible_indices = [i for i, s in enumerate(sentences) if len(s.split()) >= 4]
    eligible_sentences = [sentences[i] for i in eligible_indices]

    needs_cite = [False] * len(sentences)
    if eligible_sentences:
        logger.info("Checking %d eligible sentences for citation need (batched)…", len(eligible_sentences))
        try:
            batch_results = _batch_needs_citation(eligible_sentences)
        except Exception as e:
            # Abort rather than guess. A misaligned verdict list attributes one
            # sentence's decision to another, and nothing has been written yet,
            # so stopping here costs no work and avoids a misleading draft.
            logger.error(
                "Citation-need check failed after retries: %s\n"
                "  → Aborting without writing output. Re-run to try again, or "
                "shorten the draft if the model keeps truncating its reply.", e,
            )
            return None
        for idx, needs in zip(eligible_indices, batch_results):
            needs_cite[idx] = needs

    # ── Process sentences ────────────────────────────────────────────────
    cited_sentences = []
    key_registry = {}      # every source offered to the model: citation → cite_N
    citation_entries = []  # For the report
    next_cite_idx = 1

    for i, sentence in enumerate(sentences):
        logger.info("[%d/%d] %s", i + 1, len(sentences), sentence[:80])
        entry = {"original": sentence, "cited": False}

        if not needs_cite[i]:
            reason = "too short" if len(sentence.split()) < 4 else "no citation needed"
            logger.info(" -> %s, skipping.", reason)
            cited_sentences.append(sentence)
            entry["skip_reason"] = reason
            citation_entries.append(entry)
            continue

        logger.info(" -> Needs citation. Searching context…")
        results = hybrid_search(sentence, collection, bm25, texts, metadatas, top_k=3)

        if not results:
            logger.info(" -> No context found.")
            cited_sentences.append(sentence)
            entry["skip_reason"] = "no relevant context found in database"
            citation_entries.append(entry)
            continue

        context_str = ""
        candidates = []
        for r in results:
            cit_source = r["metadata"].get("citation_source", "Unknown")
            # Keys are registered for every retrieved chunk so the model has a
            # stable label to reference, but registration is NOT the same as
            # use — the final mapping is filtered down to keys that actually
            # made it into the draft (see below).
            if cit_source not in key_registry:
                key_registry[cit_source] = f"cite_{next_cite_idx}"
                next_cite_idx += 1

            cite_key = key_registry[cit_source]
            context_str += f"--- Context (Cite Key: {cite_key}) ---\n{r['text']}\n\n"
            candidates.append({"key": cite_key, "citation": cit_source})

        try:
            cited_sentence, reasoning = _cite_sentence_with_reasoning(sentence, context_str)
            cited_sentences.append(cited_sentence)

            # A successful call is not a citation. When the model declines, or
            # when the response could not be parsed, _cite_sentence_with_reasoning
            # returns the original sentence unchanged — so decide from the text.
            if _cite_keys(cited_sentence):
                logger.info(" -> Cited: %s", cited_sentence[:80])
                entry["cited"] = True
                entry["cited_text"] = cited_sentence
                entry["reasoning"] = reasoning
                entry["sources"] = candidates
            else:
                logger.info(" -> Declined: retrieved context did not support the claim.")
                entry["skip_reason"] = "context retrieved but the model did not cite it"
                entry["reasoning"] = reasoning
                entry["candidates"] = candidates
        except Exception as e:
            logger.error(" -> Error during citing: %s", e)
            cited_sentences.append(sentence)
            entry["skip_reason"] = f"LLM error: {e}"

        citation_entries.append(entry)

    # ── Write outputs ────────────────────────────────────────────────────
    final_draft = " ".join(cited_sentences)

    with open(out_path, "w") as f:
        f.write(final_draft)
    logger.info("Saved cited draft to %s", out_path)

    # The mapping is the BibTeX input, so it must describe the draft as written.
    # key_registry holds every source that was offered to the model; only the
    # keys the model actually used belong here.
    used_keys = _cite_keys(final_draft)
    citation_mapping = {
        source: key for source, key in key_registry.items() if key in used_keys
    }

    unknown = used_keys - set(key_registry.values())
    if unknown:
        logger.warning(
            "Draft cites %d key(s) that were never offered as context — the model "
            "likely invented them: %s",
            len(unknown), ", ".join(sorted(unknown)),
        )

    logger.info(
        "%d of %d retrieved sources were actually cited.",
        len(citation_mapping), len(key_registry),
    )

    mapping_file = out_path.replace(".txt", "_citations.json")
    with open(mapping_file, "w") as f:
        json.dump(citation_mapping, f, indent=4)
    logger.info("Saved citation mapping to %s", mapping_file)

    # ── Generate citation reasoning report ───────────────────────────────
    report_path = out_path.replace(".txt", "_report.md")
    _generate_report(
        file_path, report_path, sentences, cited_sentences,
        needs_cite, citation_entries, citation_mapping, key_registry,
    )

    # Returned so callers can tell a completed run from an aborted one. The
    # early returns above yield None; only this path wrote an output file.
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Citation Agent")
    parser.add_argument("--file", type=str, required=True, help="Path to the draft text file.")
    parser.add_argument("--out", type=str, default="cited_draft.txt", help="Path to save the cited draft.")
    args = parser.parse_args()

    run_batch_citer(args.file, args.out)
