import json
import os
import requests
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET

def fetch_papers():
    with open("extracted_citations.json", "r") as f:
        citations = json.load(f)
        
    citations = [c for c in citations if len(c) < 500]
    print(f"Loaded {len(citations)} valid citations to fetch.")
    
    os.makedirs("pulled_pdfs", exist_ok=True)
    
    downloaded = {}
    failed = []
    
    email = "researcher123987@gmail.com"
    
    for i, citation in enumerate(citations):
        print(f"[{i+1}/{len(citations)}] Processing: {citation[:60]}...")
        pdf_saved = False
        fail_reason = ""
        
        try:
            # 1. Query Crossref to find DOI
            crossref_url = "https://api.crossref.org/works"
            params = {"query.bibliographic": citation, "rows": 1, "select": "DOI,title"}
            res = requests.get(crossref_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            doi = None
            title = "Unknown"
            
            if data["message"]["items"]:
                item = data["message"]["items"][0]
                doi = item.get("DOI")
                title = item.get("title", ["Unknown"])[0]
                
            if doi:
                # 2. Query Unpaywall using the DOI
                unpaywall_url = f"https://api.unpaywall.org/v2/{doi}"
                u_params = {"email": email}
                u_res = requests.get(unpaywall_url, params=u_params, timeout=10)
                
                if u_res.status_code == 200:
                    u_data = u_res.json()
                    if u_data.get("is_oa") and u_data.get("best_oa_location"):
                        pdf_url = u_data["best_oa_location"].get("url_for_pdf")
                        if pdf_url:
                            # 3. Download the PDF
                            pdf_res = requests.get(pdf_url, stream=True, timeout=15)
                            if pdf_res.status_code == 200:
                                safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:50]
                                filename = f"pulled_pdfs/{safe_title}_{i}.pdf"
                                with open(filename, "wb") as pdf_file:
                                    for chunk in pdf_res.iter_content(chunk_size=8192):
                                        pdf_file.write(chunk)
                                downloaded[filename] = citation
                                print(f" --> SUCCESS! Saved to {filename}")
                                pdf_saved = True
                            else:
                                fail_reason = f"PDF download link failed (status {pdf_res.status_code})"
                        else:
                            fail_reason = "Open Access, but no direct PDF URL"
                    else:
                        fail_reason = "Paywalled / Not Open Access"
                else:
                    fail_reason = "Unpaywall lookup failed"
            else:
                fail_reason = "No DOI found in Crossref"

            # 4. Fallback to arXiv if Unpaywall failed or NO DOI
            if not pdf_saved:
                print(f"     -> Unpaywall missed it ({fail_reason}). Trying arXiv fallback...")
                query = f'ti:"{title}"' if title != "Unknown" else f'all:"{citation}"'
                safe_query = urllib.parse.quote(query)
                arxiv_url = f'http://export.arxiv.org/api/query?search_query={safe_query}&max_results=1'
                
                arxiv_res = requests.get(arxiv_url, timeout=10)
                if arxiv_res.status_code == 200:
                    root = ET.fromstring(arxiv_res.content)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    entries = root.findall('atom:entry', ns)
                    
                    if entries:
                        entry = entries[0]
                        pdf_link = None
                        for link in entry.findall('atom:link', ns):
                            if link.attrib.get('title') == 'pdf':
                                pdf_link = link.attrib.get('href')
                                break
                                
                        if pdf_link:
                            pdf_link = pdf_link.replace("http://", "https://")
                            if not pdf_link.endswith('.pdf'):
                                pdf_link += '.pdf'
                                
                            a_res = requests.get(pdf_link, stream=True, timeout=15)
                            if a_res.status_code == 200:
                                safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:50]
                                filename = f"pulled_pdfs/{safe_title}_{i}_arxiv.pdf"
                                with open(filename, "wb") as pdf_file:
                                    for chunk in a_res.iter_content(chunk_size=8192):
                                        pdf_file.write(chunk)
                                        
                                downloaded[filename] = citation
                                print(f" --> SUCCESS via arXiv! Saved to {filename}")
                                pdf_saved = True
                                time.sleep(3) # arXiv rate limits (1 req / 3 sec)
                            else:
                                fail_reason = f"{fail_reason} | arXiv PDF download failed"
                        else:
                            fail_reason = f"{fail_reason} | No PDF link on arXiv"
                    else:
                        fail_reason = f"{fail_reason} | Not found on arXiv"
                else:
                    fail_reason = f"{fail_reason} | arXiv API error"
            
            if not pdf_saved:
                failed.append({"citation": citation, "reason": fail_reason or "Fetch failed"})

        except Exception as e:
            failed.append({"citation": citation, "reason": str(e)})
            print(f" --> ERROR: {e}")
            
        if pdf_saved:
            time.sleep(0.5) # Courtesy sleep for unpaywall
        else:
            time.sleep(3) # if we fell through to arxiv, ensure we pause

    # Write metadata
    with open("downloaded.json", "w") as f:
        json.dump(downloaded, f, indent=4)
        
    with open("failed_downloads.json", "w") as f:
        json.dump(failed, f, indent=4)
        
    print(f"\nDone! Downloaded {len(downloaded)}, Failed {len(failed)}")

if __name__ == "__main__":
    fetch_papers()
