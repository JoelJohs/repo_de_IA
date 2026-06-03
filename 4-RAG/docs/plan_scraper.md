# PDF Scraper for Security RAG Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a robust, modular Python scraper to find, validate (min 5 pages, 2018-2026), and organize high-quality PDFs into a multi-category corpus for a RAG system on public security in Mexico.

**Architecture:** A modular class-based script (`scraper.py`) using a "Deep Inventory" approach (probing PDFs before full download) and a semi-automated workflow (Search -> Review -> Download).

**Tech Stack:** Python 3.10+, `requests`, `PyMuPDF` (fitz), `googlesearch-python`.

---

### Task 1: Environment & Directory Setup

**Files:**
- Create: `IA/4-RAG/src/ingest/scraper.py`
- Modify: `IA/4-RAG/requirements.txt` (if exists) or just install directly.

- [ ] **Step 1: Create directory structure**
Run: `mkdir -p IA/4-RAG/corpus/metadata IA/4-RAG/src/ingest`

- [ ] **Step 2: Install dependencies**
Run: `pip install requests pymupdf googlesearch-python`

- [ ] **Step 3: Create script skeleton**
Write basic imports and class structure in `IA/4-RAG/src/ingest/scraper.py`.

```python
import os
import json
import time
import random
import requests
import fitz  # PyMuPDF
from googlesearch import search

class CorpusScraper:
    def __init__(self, base_dir="IA/4-RAG/corpus"):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata", "documentos.json")
        self.categories = {
            "homicidios": ["homicidios dolosos", "homicidio doloso"],
            "crimen_organizado": ["crimen organizado", "Michoacán crimen organizado", "Guerrero crimen organizado"],
            "tierra_caliente": ["Tierra Caliente"],
            "desplazamiento_forzado": ["desplazamiento forzado interno", "desplazamiento interno"],
            "extorsion": ["extorsión", "cobro de piso"],
            "militarizacion": ["militarización seguridad pública"],
            "prevencion_social": ["prevención social de la violencia"],
            "violencia_urbana": ["violencia urbana", "impacto socioeconómico de la violencia"],
            "violencia_rural": ["violencia rural"],
            "registro_victimas": ["registro de víctimas", "estadísticas oficiales de violencia"],
            "cifra_negra": ["cifra negra"],
            "subregistro": ["subregistro delitos"]
        }
```

- [ ] **Step 4: Commit**
```bash
git add IA/4-RAG/src/ingest/scraper.py
git commit -m "feat: initial scraper structure and directory setup"
```

---

### Task 2: Implement the Searcher Module

**Files:**
- Modify: `IA/4-RAG/src/ingest/scraper.py`

- [ ] **Step 1: Implement search logic with rate limiting**
Add `perform_search` method to handle queries and return a list of candidate URLs.

```python
    def perform_search(self, query, num_results=10):
        print(f"[*] Searching for: {query}")
        results = []
        try:
            for url in search(f"{query} filetype:pdf", num=num_results, stop=num_results, pause=random.uniform(5, 10)):
                results.append(url)
        except Exception as e:
            print(f"[!] Search error: {e}")
        return results
```

- [ ] **Step 2: Implement query generator**
Create a method to iterate through all topics and collect URLs.

- [ ] **Step 3: Commit**
```bash
git commit -m "feat: add searcher module with rate limiting"
```

---

### Task 3: Implement the Prober Module (Deep Inventory)

**Files:**
- Modify: `IA/4-RAG/src/ingest/scraper.py`

- [ ] **Step 1: Implement partial PDF download (Probing)**
Use `Range` header to get first 256KB and check page count/metadata.

```python
    def probe_pdf(self, url):
        try:
            headers = {"Range": "bytes=0-262144", "User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=15, stream=True)
            if response.status_code in [200, 206]:
                # Temporary save for parsing
                with open("temp.pdf", "wb") as f:
                    f.write(response.content)
                
                doc = fitz.open("temp.pdf")
                pages = doc.page_count
                meta = doc.metadata
                doc.close()
                os.remove("temp.pdf")
                
                return {
                    "pages": pages,
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "valid": pages >= 5
                }
        except Exception as e:
            print(f"[!] Probing error for {url}: {e}")
        return None
```

- [ ] **Step 2: Commit**
```bash
git commit -m "feat: add prober module for PDF metadata extraction"
```

---

### Task 4: Implement Inventor & JSON Persistence

**Files:**
- Modify: `IA/4-RAG/src/ingest/scraper.py`

- [ ] **Step 1: Implement metadata management**
Add methods to load/save `documentos.json` and avoid duplicates.

- [ ] **Step 2: Implement topic-to-category mapping**
Ensure documents are assigned to the correct output folders.

- [ ] **Step 3: Commit**
```bash
git commit -m "feat: add inventory management and JSON persistence"
```

---

### Task 5: Final CLI and Download Logic

**Files:**
- Modify: `IA/4-RAG/src/ingest/scraper.py`

- [ ] **Step 1: Implement full download logic**
Download approved documents to their respective folders.

- [ ] **Step 2: Add CLI arguments**
Use `argparse` for `--search` and `--download`.

- [ ] **Step 3: Final validation run**
Run a test search for one topic to verify the inventory creation.

- [ ] **Step 4: Commit**
```bash
git commit -m "feat: complete scraper with CLI and download logic"
```
