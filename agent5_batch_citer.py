import argparse
import json
import os
import re

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
def _cite_sentence(sentence, context_str):
    """Ask the LLM to rewrite a sentence with a LaTeX \\cite{key} tag."""
    sys_prompt = (
        "You are an expert writing assistant. Below is a sentence and some retrieved context. "
        "Your job is to append a LaTeX citation \\cite{key} to the sentence if the context supports it. "
        "You MUST use the exact 'Cite Key' provided in the context blocks. "
        "Output ONLY the rewritten sentence, nothing else."
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
        cited = response.message.content.strip()
    except AttributeError:
        cited = response["message"]["content"].strip()

    prompt_tokens = getattr(response, "prompt_eval_count", "N/A")
    completion_tokens = getattr(response, "eval_count", "N/A")
    logger.info(
        " -> [Token Stats] Submitted: %s | Generated: %s", prompt_tokens, completion_tokens
    )
    return cited


def run_batch_citer(file_path, out_path="cited_draft.txt"):
    if not os.path.exists(file_path):
        logger.error("File %s not found.", file_path)
        return

    with open(file_path, "r") as f:
        draft_text = f.read()

    collection, bm25, texts, metadatas = load_search_resources()

    sentences = split_into_sentences(draft_text)
    logger.info("Split draft into %d sentences.", len(sentences))

    # ── Batch citation-need check (item #14) ─────────────────────────────
    # Filter out very short sentences first
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
    next_cite_idx = 1

    for i, sentence in enumerate(sentences):
        logger.info("[%d/%d] %s", i + 1, len(sentences), sentence[:80])

        if not needs_cite[i]:
            reason = "too short" if len(sentence.split()) < 4 else "no citation needed"
            logger.info(" -> %s, skipping.", reason)
            cited_sentences.append(sentence)
            continue

        logger.info(" -> Needs citation. Searching context…")
        results = hybrid_search(sentence, collection, bm25, texts, metadatas, top_k=3)

        if not results:
            logger.info(" -> No context found.")
            cited_sentences.append(sentence)
            continue

        context_str = ""
        for r in results:
            cit_source = r["metadata"].get("citation_source", "Unknown")
            if cit_source not in citation_mapping:
                citation_mapping[cit_source] = f"cite_{next_cite_idx}"
                next_cite_idx += 1

            cite_key = citation_mapping[cit_source]
            context_str += f"--- Context (Cite Key: {cite_key}) ---\n{r['text']}\n\n"

        try:
            cited_sentence = _cite_sentence(sentence, context_str)
            logger.info(" -> Cited: %s", cited_sentence[:80])
            cited_sentences.append(cited_sentence)
        except Exception as e:
            logger.error(" -> Error during citing: %s", e)
            cited_sentences.append(sentence)

    final_draft = " ".join(cited_sentences)

    with open(out_path, "w") as f:
        f.write(final_draft)
    logger.info("Saved cited draft to %s", out_path)

    mapping_file = out_path.replace(".txt", "_citations.json")
    with open(mapping_file, "w") as f:
        json.dump(citation_mapping, f, indent=4)
    logger.info("Saved citation mapping to %s", mapping_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Citation Agent")
    parser.add_argument("--file", type=str, required=True, help="Path to the draft text file.")
    parser.add_argument("--out", type=str, default="cited_draft.txt", help="Path to save the cited draft.")
    args = parser.parse_args()

    run_batch_citer(args.file, args.out)
