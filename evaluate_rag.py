import os
import pickle
import chromadb
import pandas as pd
from datasets import Dataset
import ollama
import re

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.run_config import RunConfig
# Use wrapper if ragas version requires it
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from agent4_assistant import hybrid_search

def load_data():
    print("Connecting to Vector DB and BM25...")
    db_path = "./physics_vectordb"
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name="physics_papers")

    paired_data = []
    limit = 1000
    offset = 0
    while True:
        batch = collection.get(include=["documents", "metadatas"], limit=limit, offset=offset)
        if not batch["ids"]:
            break
        for doc, meta, chunk_id in zip(batch["documents"], batch["metadatas"], batch["ids"]):
            try:
                idx = int(chunk_id.split("_")[1])
                paired_data.append((idx, doc, meta))
            except:
                pass
        offset += limit

    paired_data.sort(key=lambda x: x[0])
    texts = [item[1] for item in paired_data]
    metadatas = [item[2] for item in paired_data]

    with open("./bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return collection, bm25, texts, metadatas

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

    system_prompt = (
        "You are an academic writing assistant specializing in physics. "
        "The user will provide a snippet of text they are writing. "
        "I will provide retrieved scientific context and the precise formal citations those contexts belong to. "
        "Your task is to rewrite the user snippet inserting the correct citation where structurally appropriate using LaTeX format, "
        "and explain why that specific citation supports their writing."
    )

    user_prompt = f"User Draft Text:\n{query}\n\nRetrieved Context & Formal Citations:\n{context_str}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = ollama.chat(
        model="gemma4:latest",
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
    print("Loading RAG pipeline data...")
    collection, bm25, texts, metadatas = load_data()
    
    print("Parsing sample inputs...")
    queries = parse_inputs("sample_inputs")
    print(f"Found {len(queries)} queries to evaluate.")
    
    data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
    }
    
    print("\n--- Generating Responses ---")
    for idx, query in enumerate(queries):
        print(f"\nProcessing query {idx+1}/{len(queries)}...")
        answer, contexts = get_rag_response(query, collection, bm25, texts, metadatas)
        data["user_input"].append(query)
        data["response"].append(answer)
        data["retrieved_contexts"].append(contexts)
        print(f"Retrieved {len(contexts)} contexts.")
        
    dataset = Dataset.from_dict(data)
    
    print("\n--- Starting Ragas Evaluation ---")
    print("Initializing evaluator LLM: deepseek-r1:14b")
    # Wrap with Ragas wrappers
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="deepseek-r1:14b", temperature=0.0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text:latest"))
    
    run_config = RunConfig(timeout=1800, max_retries=5)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )
    
    print("\n--- Evaluation Results ---")
    print(results)
    
    # Save results to CSV
    df = results.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("Detailed results saved to 'evaluation_results.csv'.")

if __name__ == "__main__":
    main()
