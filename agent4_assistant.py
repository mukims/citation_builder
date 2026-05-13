import argparse
import pickle
import chromadb
import numpy as np
import ollama
import re

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    print("Please run this in your rag_prod environment.")
    exit(1)

def hybrid_search(query, collection, bm25, texts, metadatas, top_k=3, rrf_k=60):
    k_cand = max(15, top_k * 3)

    # 1. Sparse (BM25) retrieval
    tokenized_query = re.findall(r'\w+', query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_top_indices = np.argsort(bm25_scores)[::-1][:k_cand]

    # 2. Dense (embedding) retrieval
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    query_emb = embeddings_model.embed_query(query)

    dense_results = collection.query(
        query_embeddings=[query_emb], 
        n_results=k_cand,
        include=["documents", "metadatas", "distances"]
    )

    dense_ids_ordered = []
    if dense_results["ids"] and dense_results["ids"][0]:
        for id_str in dense_results["ids"][0]:
            try:
                dense_ids_ordered.append(int(id_str.split("_")[1]))
            except:
                pass

    dense_distances = dense_results.get("distances", [[]])[0]

    # 3. Reciprocal Rank Fusion (RRF)
    fused_scores = {}

    for rank, idx in enumerate(sparse_top_indices):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rank + 1 + rrf_k))

    for rank, idx in enumerate(dense_ids_ordered):
        fused_scores[idx] = fused_scores.get(idx, 0.0) + (1.0 / (rank + 1 + rrf_k))

    # 4. Rank and return top_k
    ranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, rrf_score in ranked_results:
        # Avoid index errors if database changed between BM25 creation and now
        if idx < len(texts):
            results.append({
                "chunk_index": idx,
                "text": texts[idx],
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "rrf_score": round(rrf_score, 5)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Citation AI Assistant")
    parser.add_argument("--text", type=str, required=True, help="Draft text you want to cite.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved components.")
    args = parser.parse_args()

    print("Connecting to Vector DB and BM25...")
    db_path = "./physics_vectordb"
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name="physics_papers")

    # Fetch mapping for BM25 alignment
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

    print(f"Searching DB for relevant context for query: '{args.text}'")
    results = hybrid_search(args.text, collection, bm25, texts, metadatas, top_k=args.top_k)

    if not results:
        print("No related passages found in the database.")
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
        
    print("\nDrafting citation suggestion...")
    system_prompt = (
        "You are an academic writing assistant specializing in physics. "
        "The user will provide a snippet of text they are writing. "
        "I will provide retrieved scientific context and the precise formal citations those contexts belong to. "
        "Your task is to rewrite the user snippet inserting the correct citation where structurally appropriate using LaTeX format, "
        "and explain why that specific citation supports their writing."
    )

    user_prompt = f"User Draft Text:\n{args.text}\n\nRetrieved Context & Formal Citations:\n{context}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = ollama.chat(
            model="gemma4:latest",
            messages=messages,
            stream=False
        )
        
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
        print(f"\nError communicating with Ollama: {e}")

if __name__ == "__main__":
    main()
