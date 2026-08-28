import os
import re
import pandas as pd
from datasets import Dataset
import ollama

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
# Use wrapper if ragas version requires it
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import (
    LLM_MODEL,
    EVAL_MODEL,
    EMBED_MODEL,
    SAMPLE_INPUTS_PATH,
    EVAL_RESULTS_PATH,
)
from prompts import CITATION_SUGGESTION_SYSTEM, CITATION_SUGGESTION_USER
from shared.log import get_logger
from shared.db import load_search_resources
from shared.search import hybrid_search

logger = get_logger("evaluate")


def get_rag_response(query, collection, bm25, texts, metadatas):
    results = hybrid_search(query, collection, bm25, texts, metadatas, top_k=3)
    
    context_chunks = []
    context_str = ""
    for i, r in enumerate(results):
        meta = r["metadata"]
        cit = meta.get("citation_source", "Unknown Citation")
        doc_name = meta.get("document", "Unknown Document")
        
        chunk_text = r["text"]
        context_chunks.append(chunk_text)
        
        context_str += f"--- Source {i+1} : Document '{doc_name}' corresponding to citation {cit} ---\n"
        context_str += chunk_text + "\n\n"

    system_prompt = CITATION_SUGGESTION_SYSTEM

    user_prompt = CITATION_SUGGESTION_USER.format(query=query, context=context_str)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=False
    )
    
    try:
        answer = response.message.content
    except AttributeError:
        answer = response['message']['content']

    return answer, context_chunks

def parse_inputs(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by one or more blank lines
    blocks = re.split(r'\n\s*\n', content)
    queries = [b.strip() for b in blocks if b.strip()]
    return queries

def main():
    logger.info("Loading RAG pipeline data…")
    collection, bm25, texts, metadatas = load_search_resources()
    
    logger.info("Parsing sample inputs…")
    queries = parse_inputs(SAMPLE_INPUTS_PATH)
    logger.info("Found %d queries to evaluate.", len(queries))
    
    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
    }
    
    logger.info("--- Generating Responses ---")
    for idx, query in enumerate(queries):
        logger.info("Processing query %d/%d…", idx + 1, len(queries))
        answer, contexts = get_rag_response(query, collection, bm25, texts, metadatas)
        data["user_input"].append(query)
        data["response"].append(answer)
        data["retrieved_contexts"].append(contexts)
        logger.info("Retrieved %d contexts.", len(contexts))
        
    dataset = Dataset.from_dict(data)
    
    logger.info("--- Starting Ragas Evaluation ---")
    logger.info("Initializing evaluator LLM: %s", EVAL_MODEL)
    # Wrap with Ragas wrappers
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=EVAL_MODEL, temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBED_MODEL))
    
    run_config = RunConfig(timeout=1800, max_retries=5)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )
    
    logger.info("--- Evaluation Results ---")
    print(results)
    
    # Save results to CSV
    df = results.to_pandas()
    df.to_csv(EVAL_RESULTS_PATH, index=False)
    logger.info("Detailed results saved to %s", EVAL_RESULTS_PATH)

if __name__ == "__main__":
    main()
