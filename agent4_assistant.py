import argparse
import ollama

from config import LLM_MODEL
from shared.log import get_logger
from shared.db import load_search_resources
from shared.search import hybrid_search
from shared.retry import retry

logger = get_logger("agent4")


@retry(max_retries=3, backoff=2.0)
def _generate_suggestion(system_prompt, user_prompt):
    """Call the LLM to generate a citation suggestion (with retry)."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
    )
    return response


def main():
    parser = argparse.ArgumentParser(description="Citation AI Assistant")
    parser.add_argument("--text", type=str, required=True, help="Draft text you want to cite.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved components.")
    args = parser.parse_args()

    collection, bm25, texts, metadatas = load_search_resources()

    logger.info("Searching DB for relevant context for query: '%s'", args.text)
    results = hybrid_search(args.text, collection, bm25, texts, metadatas, top_k=args.top_k)

    if not results:
        logger.info("No related passages found in the database.")
        return

    context = ""
    citations_pool = set()
    for i, r in enumerate(results):
        meta = r["metadata"]
        cit = meta.get("citation_source", "Unknown Citation")
        doc_name = meta.get("document", "Unknown Document")
        
        citations_pool.add(cit)
        context += f"--- Source {i+1} : Document '{doc_name}' corresponding to citation {cit} ---\n"
        context += r["text"] + "\n\n"

    print("--- Found References ---")
    for cit in citations_pool:
        print(f" > {cit}")
        
    logger.info("Drafting citation suggestion…")
    system_prompt = (
        "You are an academic writing assistant specializing in physics. "
        "The user will provide a snippet of text they are writing. "
        "I will provide retrieved scientific context and the precise formal citations those contexts belong to. "
        "Your task is to rewrite the user snippet inserting the correct citation where structurally appropriate using LaTeX format, "
        "and explain why that specific citation supports their writing."
    )

    user_prompt = f"User Draft Text:\n{args.text}\n\nRetrieved Context & Formal Citations:\n{context}"
    
    try:
        response = _generate_suggestion(system_prompt, user_prompt)
        
        prompt_tokens = getattr(response, 'prompt_eval_count', 'N/A')
        completion_tokens = getattr(response, 'eval_count', 'N/A')
        print(f"\n[Token Stats] Submitted: {prompt_tokens} | Generated: {completion_tokens}")

        print("\n=== AI ASSISTANT SUGGESTION ===")
        try:
            print(response.message.content)
        except AttributeError:
            print(response['message']['content'])
        print("===============================\n")
    except Exception as e:
        logger.error("Error communicating with Ollama: %s", e)

if __name__ == "__main__":
    main()
