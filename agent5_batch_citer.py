import argparse
import json
import os
import re
from datetime import datetime

import ollama

from config import LLM_MODEL
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
    # Split on sentence-ending punctuation NOT preceded by known abbreviations
    pattern = rf'(?<!{_ABBREVS})(?<=[.!?])\s+'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ─── Batched citation-need check ─────────────────────────────────────────────

@retry(max_retries=2, backoff=2.0)
def _batch_needs_citation(sentences):
    """
    Ask the LLM once whether each sentence in a batch needs a citation.
    Returns a list of booleans aligned with the input list.
    """
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = (
        "Below is a numbered list of sentences from an academic draft.\n"
        "For EACH sentence, decide whether it states a factual scientific claim "
        "that requires a citation.\n"
        "Reply with ONLY a numbered list of YES or NO, one per line. Example:\n"
        "1. YES\n2. NO\n3. YES\n\n"
        f"Sentences:\n{numbered}"
    )
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        answer = response.message.content
    except AttributeError:
        answer = response["message"]["content"]

    # Parse the YES/NO list
    results = []
    for line in answer.strip().split("\n"):
        line = line.strip().upper()
        results.append("YES" in line)

    # Pad or truncate to match input length
    while len(results) < len(sentences):
        results.append(False)
    return results[: len(sentences)]


@retry(max_retries=3, backoff=2.0)
def _cite_sentence_with_reasoning(sentence, context_str):
    """Ask the LLM to rewrite a sentence with \\cite{key} and explain why.

    Returns:
        tuple: (cited_sentence, reasoning)
    """
    sys_prompt = (
        "You are an expert writing assistant. Below is a sentence and some retrieved context. "
        "Your job is:\n"
        "1. Rewrite the sentence by appending a LaTeX citation \\cite{key} if the context supports it. "
        "You MUST use the exact 'Cite Key' provided in the context blocks.\n"
        "2. Provide a brief explanation (2-3 sentences) of WHY this citation is appropriate — "
        "what specific claim in the sentence is supported by the source.\n\n"
        "Format your response EXACTLY like this:\n"
        "CITED: <the rewritten sentence with \\cite{key}>\n"
        "REASON: <2-3 sentence justification>"
    )
    user_prompt = f"Sentence: {sentence}\n\nRetrieved Context:\n{context_str}"

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
):
    """Generate a markdown report explaining every citation decision."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    basename = os.path.basename(file_path)

    lines = [
        f"# Citation Report — `{basename}`",
        f"*Generated on {timestamp}*\n",
        "---\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total sentences | {len(sentences)} |",
        f"| Cited sentences | {sum(1 for e in citation_entries if e.get('cited'))} |",
        f"| Skipped (no citation needed) | {sum(1 for n in needs_cite if not n)} |",
        f"| Unique sources used | {len(citation_mapping)} |",
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
            lines.append(f"**Decision:** No citation needed — {entry.get('skip_reason', 'skipped')}\n")
        else:
            lines.append("**Original:**")
            lines.append(f"> {entry['original']}\n")
            lines.append("**With citation:**")
            lines.append(f"> {entry['cited']}\n")

            if entry.get("reasoning"):
                lines.append("**Reasoning:**")
                lines.append(f"{entry['reasoning']}\n")

            if entry.get("sources"):
                lines.append("**Sources used:**")
                for src in entry["sources"]:
                    lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                lines.append("")

        lines.append("---\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Saved citation report to %s", report_path)


def run_batch_citer(file_path, out_path="cited_draft.txt"):
    if not os.path.exists(file_path):
        logger.error("File %s not found.", file_path)
        return

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
        batch_results = _batch_needs_citation(eligible_sentences)
        for idx, needs in zip(eligible_indices, batch_results):
            needs_cite[idx] = needs

    # ── Process sentences ────────────────────────────────────────────────
    cited_sentences = []
    citation_mapping = {}
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
        sources_used = []
        for r in results:
            cit_source = r["metadata"].get("citation_source", "Unknown")
            if cit_source not in citation_mapping:
                citation_mapping[cit_source] = f"cite_{next_cite_idx}"
                next_cite_idx += 1

            cite_key = citation_mapping[cit_source]
            context_str += f"--- Context (Cite Key: {cite_key}) ---\n{r['text']}\n\n"
            sources_used.append({"key": cite_key, "citation": cit_source})

        try:
            cited_sentence, reasoning = _cite_sentence_with_reasoning(sentence, context_str)
            logger.info(" -> Cited: %s", cited_sentence[:80])
            cited_sentences.append(cited_sentence)

            entry["cited"] = True
            entry["cited_text"] = cited_sentence
            entry["reasoning"] = reasoning
            entry["sources"] = sources_used
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

    mapping_file = out_path.replace(".txt", "_citations.json")
    with open(mapping_file, "w") as f:
        json.dump(citation_mapping, f, indent=4)
    logger.info("Saved citation mapping to %s", mapping_file)

    # ── Generate citation reasoning report ───────────────────────────────
    report_path = out_path.replace(".txt", "_report.md")
    _generate_report(
        file_path, report_path, sentences, cited_sentences,
        needs_cite, citation_entries, citation_mapping,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Citation Agent")
    parser.add_argument("--file", type=str, required=True, help="Path to the draft text file.")
    parser.add_argument("--out", type=str, default="cited_draft.txt", help="Path to save the cited draft.")
    args = parser.parse_args()

    run_batch_citer(args.file, args.out)
