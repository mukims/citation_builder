import os
import json
import re
import cv2
import fitz
import numpy as np
import ollama
from tqdm import tqdm
from pathlib import Path
import chromadb
import pickle
from rank_bm25 import BM25Okapi
import layoutparser as lp
import argparse
import concurrent.futures
import multiprocessing

try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError as e:
    print(f"Failed to import LangChain modules. Error: {e}")
    exit(1)

def find_cap(text_blocks, bbox, box_type):
    x1, y1, x2, y2 = bbox 
    best, min_dist = "", float("inf")

    for b in text_blocks:
        tx0, ty0, tx1, ty1 = b['bbox']
        text = b['text']

        horiz_overlap = max(0, min(x2, tx1) - max(x1, tx0))
        if horiz_overlap < 10 and (x2 - x1) > 100:
            continue

        if box_type == "Figure":
            dist = ty0 - y2  # distance below the figure
            if -20 < dist < 400:
                score = dist - (200 if text.lower().startswith("fig") else 0)
                if score < min_dist:
                    min_dist, best = score, text

    return best

def clean_text(text):
    text = re.sub(r'\b(\w) (\w) ', r'\1\2', text)
    for _ in range(20):
        text = re.sub(r'(?<!\w)(\w) (\w)(?!\w)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_pdf_file(pdf_path, citation_string, detectron_weights, images_dir):
    print(f"Processing pages for {pdf_path}...")
    corpus = []
    dpi = 200
    zoom = dpi / 72.0
    
    # Init model inside worker for CUDA compatibility
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        model_path=detectron_weights,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )

    try:
        pdf = fitz.open(pdf_path)
        doc_name = os.path.basename(pdf_path)
        pdf_name = doc_name.strip().replace(" ","_").lower()
        
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            layout = model.detect(img)
            raw_blocks = page.get_text("blocks")
            text_blovk_img = []

            for b in raw_blocks:
                if b[6] == 0:
                    tx1, ty1, tx2, ty2 = [c * zoom for c in b[:4]]
                    text = b[4].replace('\n', ' ')
                    text_blovk_img.append({
                        'bbox': (tx1, ty1, tx2, ty2),
                        'text': text.strip()
                    })
                    if len(text) > 4:
                        corpus.append({
                            "document": pdf_name,
                            "citation": citation_string,
                            "page": page_idx,
                            "type": "text",
                            "content": text
                        })

            figures = [b for b in layout if b.type in ["Figure", "Table"]]

            for i, fig in enumerate(figures):
                pad = 20
                x1, y1, x2, y2 = fig.coordinates
                x1 = max(0, int(x1 - pad))
                y1 = max(0, int(y1 - pad))
                x2 = min(img.shape[1], int(x2 + pad))
                y2 = min(img.shape[0], int(y2 + pad))
                            
                cropped_img = img[y1:y2, x1:x2]
                out_name = f"{pdf_name}_p{page_idx}_f{i}.png"
                out_path = os.path.join(images_dir, out_name)
                cv2.imwrite(out_path, cropped_img)
                            
                context = find_cap(text_blovk_img, (x1, y1, x2, y2), fig.type)

                try:
                    formatted_prompt = f"You are analysing scientific plots. Describe this {fig.type.lower()}. Extract textual information, data and trends.\n\nSurrounding Document Context:\n{context}. Answer in 3-5 sentences at max.  "
                    response = ollama.chat(
                        model="gemma4:latest",
                        messages=[{
                            'role': 'user',
                            'content': formatted_prompt,
                            'images': [out_path] 
                        }]
                    )
                    try:
                        vlm_description = response.message.content
                    except AttributeError:
                        vlm_description = response['message']['content']
                except Exception as e:
                    print(f"Ollama failed on {out_name}: {e}")
                    vlm_description = "Description generation failed."

                corpus.append({
                    "document": pdf_name,
                    "citation": citation_string,
                    "page": page_idx,
                    "type": fig.type.lower(),
                    "content": vlm_description,
                    "metadata": {
                        "image_path": out_name,
                        "caption": context
                    }
                })
    except Exception as e:
        print(f"Failed to load or process {pdf_path}. Skipping. Error: {e}")
    
    return corpus


def run_ingestor(workers=1):
    with open("downloaded.json", "r") as f:
        downloaded = json.load(f)
        
    detectron_weights = os.path.abspath("../model_final.pth")
    images_dir = "../extracted_data/images"
    os.makedirs(images_dir, exist_ok=True)
    
    corpus = []
    
    if workers <= 1:
        print("Running sequentially (1 worker)...")
        for pdf_path, citation_string in downloaded.items():
            if not os.path.exists(pdf_path):
                continue
            corpus.extend(process_pdf_file(pdf_path, citation_string, detectron_weights, images_dir))
    else:
        print(f"Running in parallel with {workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for pdf_path, citation_string in downloaded.items():
                if not os.path.exists(pdf_path):
                    continue
                futures.append(executor.submit(process_pdf_file, pdf_path, citation_string, detectron_weights, images_dir))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs"):
                try:
                    local_corpus = future.result()
                    corpus.extend(local_corpus)
                except Exception as e:
                    print(f"Worker failed: {e}")

    print("Initializing Chunking and Embedding Database...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
    chroma_client = chromadb.PersistentClient(path="./physics_vectordb")
    collection = chroma_client.get_or_create_collection(
        name="physics_papers",
        metadata={"hnsw:space": "cosine"}
    )
    
    total_corpus = []
    
    # Process corpus for SemanticChunking
    for entry in corpus:
        if entry['type'] in ["figure","table"]:
            total_corpus.append(entry)
        elif entry['type'] == "text":
            content = clean_text(entry['content'])
            if len(content) < 10:
                continue
            try:
                docs = chunker.create_documents([content])
                for i, doc in enumerate(docs):
                    total_corpus.append({
                        "document": entry['document'],
                        "citation": entry['citation'],
                        "page": entry['page'],
                        "type": "text_chunk",
                        "content": doc.page_content,
                        "metadata": {
                            "original_text": content,
                            "chunk_index": i
                        }
                    })
            except Exception as e:
                print(f"Chunker failed: {e}")
                
    documents = []
    metadatas = []
    ids = []
    seen_texts = set()
    
    max_idx = -1
    limit = 1000
    offset = 0
    while True:
        batch = collection.get(limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for cid in batch["ids"]:
            try:
                idx = int(cid.split("_")[1])
                max_idx = max(max_idx, idx)
            except:
                pass
        offset += limit
    current_index = max_idx + 1

    for entry in total_corpus:
        content = entry["content"].strip()
        if content in seen_texts or len(content) < 10:
            continue
        seen_texts.add(content)
        documents.append(content)
        meta = {
            "document": entry["document"],
            "page": entry["page"],
            "type": entry["type"],
            "citation_source": entry["citation"]
        }
        if "metadata" in entry:
            for k, v in entry["metadata"].items():
                meta[f"extra_{k}"] = str(v)
        
        metadatas.append(meta)
        ids.append(f"chunk_{current_index}")
        current_index += 1

    if documents:
        print(f"Generating embeddings for {len(documents)} logic chunks...")
        batch_size = 1000
        for i in tqdm(range(0, len(documents), batch_size), desc="Ingesting Batches"):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_docs = [doc[:4000] for doc in batch_docs]
            batch_embeddings = embeddings.embed_documents(batch_docs)
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
        print("Success! Appended chunks to vector database.")
    else:
        print("No new chunks to insert.")

    print("Fetching ALL chunks to rebuild BM25 Index...")
    paired_data = []
    limit = 1000
    offset = 0
    while True:
        batch = collection.get(include=["documents"], limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for doc, chunk_id in zip(batch["documents"], batch["ids"]):
            try:
                idx = int(chunk_id.split("_")[1])
                paired_data.append((idx, doc))
            except:
                pass
        offset += limit

    paired_data.sort(key=lambda x: x[0])
    texts = [item[1] for item in paired_data]
    tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in texts]

    bm25 = BM25Okapi(tokenized_corpus)
    with open("./bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("Success! BM25 index rebuilt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestor with Parallelization")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers. Use 1 to disable parallelization, 2+ for multiprocessing.")
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)
    run_ingestor(workers=args.workers)
