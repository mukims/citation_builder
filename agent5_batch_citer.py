import argparse
import pickle
import chromadb
import ollama
import re
import json
import os
from agent4_assistant import hybrid_search

def split_into_sentences(text):
    # Basic regex sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def needs_citation(sentence):
    prompt = f"Does the following sentence state a factual scientific claim that requires a citation? Answer with only 'YES' or 'NO'.\n\nSentence: {sentence}"
    try:
        response = ollama.chat(model="gemma4:latest", messages=[{"role": "user", "content": prompt}])
        try:
            ans = response.message.content.upper()
        except AttributeError:
            ans = response['message']['content'].upper()
        return "YES" in ans
    except Exception as e:
        print(f"Error checking if citation needed: {e}")
        return False

def run_batch_citer(file_path, out_path="cited_draft.txt"):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, "r") as f:
        draft_text = f.read()

    print("Loading Databases...")
    db_path = "./physics_vectordb"
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name="physics_papers")

    # Fetch mapping for BM25
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

    sentences = split_into_sentences(draft_text)
    cited_sentences = []
    
    citation_mapping = {}
    next_cite_idx = 1

    print(f"Processing {len(sentences)} sentences...")
    for i, sentence in enumerate(sentences):
        print(f"\n[{i+1}/{len(sentences)}] Analyzing: {sentence}")
        if len(sentence.split()) < 4:
            # Too short to need a citation
            print(" -> Too short, skipping.")
            cited_sentences.append(sentence)
            continue
            
        if needs_citation(sentence):
            print(" -> Needs citation. Searching context...")
            results = hybrid_search(sentence, collection, bm25, texts, metadatas, top_k=3)
            
            if not results:
                print(" -> No context found.")
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
                
            sys_prompt = "You are an expert writing assistant. Below is a sentence and some retrieved context. Your job is to append a LaTeX citation \\cite{key} to the sentence if the context supports it. You MUST use the exact 'Cite Key' provided in the context blocks. Output ONLY the rewritten sentence, nothing else."
            user_prompt = f"Sentence: {sentence}\n\nRetrieved Context:\n{context_str}"
            
            try:
                response = ollama.chat(
                    model="gemma4:latest",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                try:
                    cited_sentence = response.message.content.strip()
                except AttributeError:
                    cited_sentence = response['message']['content'].strip()
                
                prompt_tokens = getattr(response, 'prompt_eval_count', 'N/A')
                completion_tokens = getattr(response, 'eval_count', 'N/A')
                print(f" -> [Token Stats] Submitted: {prompt_tokens} | Generated: {completion_tokens}")
                
                print(f" -> Cited: {cited_sentence}")
                cited_sentences.append(cited_sentence)
            except Exception as e:
                print(f" -> Error during citing: {e}")
                cited_sentences.append(sentence)
        else:
            print(" -> No citation needed.")
            cited_sentences.append(sentence)

    final_draft = " ".join(cited_sentences)
    
    with open(out_path, "w") as f:
        f.write(final_draft)
    print(f"\nSaved cited draft to {out_path}")
    
    mapping_file = out_path.replace(".txt", "_citations.json")
    with open(mapping_file, "w") as f:
        json.dump(citation_mapping, f, indent=4)
    print(f"Saved citation mapping to {mapping_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Citation Agent")
    parser.add_argument("--file", type=str, required=True, help="Path to the draft text file.")
    parser.add_argument("--out", type=str, default="cited_draft.txt", help="Path to save the cited draft.")
    args = parser.parse_args()
    
    run_batch_citer(args.file, args.out)
