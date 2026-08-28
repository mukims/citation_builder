"""
Agent 7 — Interactive Research Assistant (agent7_research_chat.py)

A conversational RAG agent that lets researchers brainstorm, explore ideas,
and ask open-ended questions against the ingested paper database.

Unlike Agent 4 (single-shot citation helper), Agent 7 maintains a multi-turn
conversation with memory, so researchers can refine questions, follow up on
previous answers, and explore tangential ideas — all grounded in the literature.

Usage:
    python agent7_research_chat.py
    python agent7_research_chat.py --top_k 5

Commands inside the chat:
    /clear     — Reset conversation history
    /sources   — Show the sources used in the last response
    /export    — Save the full conversation to a timestamped markdown file
    /help      — Show available commands
    quit/exit  — Exit the session
"""

import argparse
import json
import os
import re
from datetime import datetime

import ollama

from config import CHAT_MODEL, CHAT_OLLAMA_OPTIONS, DRAFTS_DIR
from prompts import RESEARCH_CHAT_SYSTEM
from shared.log import get_logger
from shared.db import load_search_resources
from shared.search import hybrid_search
from shared.retry import retry

logger = get_logger("agent7")

SYSTEM_PROMPT = RESEARCH_CHAT_SYSTEM

HELP_TEXT = """
╔══════════════════════════════════════════════╗
║           Available Commands                 ║
╠══════════════════════════════════════════════╣
║  /clear    — Reset conversation history      ║
║  /sources  — Show sources from last response ║
║  /export   — Save conversation to markdown   ║
║  /help     — Show this help message          ║
║  quit/exit — End the session                 ║
╚══════════════════════════════════════════════╝
"""


class ResearchChat:
    """Multi-turn conversational RAG agent backed by the paper database."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.history = []          # list of {"role": ..., "content": ...}
        self.last_sources = []     # sources used in the most recent answer
        self.turn_count = 0

        logger.info("Loading search resources…")
        self.collection, self.bm25, self.texts, self.metadatas = load_search_resources()
        logger.info("Ready. %d chunks in database.", len(self.texts))

    # ── RAG retrieval ────────────────────────────────────────────────────

    def _retrieve_context(self, query: str) -> tuple[str, list[dict]]:
        """Run hybrid search and format the results into a context block."""
        results = hybrid_search(
            query,
            self.collection,
            self.bm25,
            self.texts,
            self.metadatas,
            top_k=self.top_k,
        )

        if not results:
            return "", []

        context_parts = []
        sources = []
        seen_citations = set()

        for i, r in enumerate(results):
            meta = r["metadata"]
            citation = meta.get("citation_source", "Unknown")
            doc_name = meta.get("document", "Unknown")
            page = meta.get("page", "?")

            context_parts.append(
                f"--- Source {i+1} | Document: {doc_name} | Page: {page} | "
                f"Citation: {citation} ---\n{r['text']}"
            )

            if citation not in seen_citations:
                seen_citations.add(citation)
                sources.append({
                    "citation": citation,
                    "document": doc_name,
                    "relevance_score": r["rrf_score"],
                })

        return "\n\n".join(context_parts), sources

    # ── LLM call ─────────────────────────────────────────────────────────

    @retry(max_retries=3, backoff=2.0)
    def _generate(self, messages: list[dict], stream: bool = False):
        """Call the LLM with the full message history."""
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            stream=stream,
            options=CHAT_OLLAMA_OPTIONS,
        )
        if not stream:
            try:
                return response.message.content
            except AttributeError:
                return response["message"]["content"]
        return response

    # ── Public API ───────────────────────────────────────────────────────

    def chat_stream(self, user_message: str):
        """Process one user turn: retrieve context, stream response, update history."""
        self.turn_count += 1

        # Retrieve relevant context from the database
        context_str, sources = self._retrieve_context(user_message)
        self.last_sources = sources

        # Build the augmented user message (context is injected per-turn so
        # the model always has fresh retrieval, but conversation history
        # provides continuity)
        if context_str:
            augmented_msg = (
                f"{user_message}\n\n"
                f"[Retrieved context from your paper database — use this to "
                f"ground your answer]\n{context_str}"
            )
        else:
            augmented_msg = (
                f"{user_message}\n\n"
                f"[No relevant context was found in the database for this query. "
                f"Answer based on general knowledge and note the limitation.]"
            )

        # Assemble messages: system + conversation history + current turn
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": augmented_msg})

        # Generate response in streaming mode
        full_answer = ""
        for chunk in self._generate(messages, stream=True):
            try:
                content = chunk.message.content
            except AttributeError:
                content = chunk["message"]["content"]
            full_answer += content
            yield content

        # Update history (store the clean user message, not the augmented one)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": full_answer})

        # Keep history manageable — trim to last 10 turns to improve latency
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def clear_history(self):
        """Reset the conversation."""
        self.history.clear()
        self.last_sources.clear()
        self.turn_count = 0

    def export_conversation(self) -> str:
        """Save the full conversation to a timestamped markdown file."""
        os.makedirs(DRAFTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(DRAFTS_DIR, f"research_chat_{timestamp}.md")

        lines = [f"# Research Chat — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        for msg in self.history:
            role = "🧑‍🔬 **Researcher**" if msg["role"] == "user" else "🤖 **Assistant**"
            lines.append(f"\n{role}\n\n{msg['content']}\n")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        return filepath


# ─── CLI ──────────────────────────────────────────────────────────────────────


def run_repl(agent, banner: str | None = None):
    """Run the interactive chat loop against *agent* until the user exits.

    Shared by this module's own CLI and the orchestrator's --chat mode. Both
    previously carried their own copy of the loop, so a command added to one
    was simply missing from the other.

    Args:
        agent:  A ready :class:`ResearchChat` instance.
        banner: Optional text printed before the first prompt.
    """
    if banner:
        print(banner)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/clear":
            agent.clear_history()
            print("✓ Conversation history cleared.\n")
            continue

        if user_input == "/sources":
            if not agent.last_sources:
                print("No sources from the last response.\n")
            else:
                print("\n📚 Sources used in the last response:")
                for i, s in enumerate(agent.last_sources, 1):
                    print(f"  {i}. [{s['document']}] {s['citation']}")
                    print(f"     Relevance: {s['relevance_score']}")
                print()
            continue

        if user_input == "/export":
            path = agent.export_conversation()
            print(f"✓ Conversation exported to {path}\n")
            continue

        try:
            print("\nAssistant: ", end="", flush=True)
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            if agent.last_sources:
                cits = {s["citation"][:60] for s in agent.last_sources[:3]}
                print(f"  📚 Drawing from: {', '.join(cits)}")
                print("  (type /sources for full list)\n")
        except Exception as e:
            logger.error("Error during chat: %s", e)
            print(f"\n⚠ Error: {e}. Please try again.\n")


def main():
    parser = argparse.ArgumentParser(description="Agent 7 — Interactive Research Assistant")
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of context chunks to retrieve per question (default: 5).",
    )
    args = parser.parse_args()

    agent = ResearchChat(top_k=args.top_k)

    run_repl(agent, banner=(
        "\n" + "=" * 60 + "\n"
        "  🔬  Research Assistant — Interactive Mode\n"
        "  Powered by your ingested paper database\n"
        "  Type /help for commands, 'quit' to exit\n"
        + "=" * 60 + "\n"
    ))


if __name__ == "__main__":
    main()
