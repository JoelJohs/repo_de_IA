#!/usr/bin/env python3
"""
Extrae texto de PDFs y divide en chunks para RAG.
Lee metadata de documentos.json, carga cada PDF con PyPDFLoader,
divide con RecursiveCharacterTextSplitter y guarda en chunks.jsonl.
"""

import json, re, sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent.parent.parent
CORPUS_PDF = BASE_DIR / "corpus" / "pdf"
METADATA_FILE = CORPUS_PDF / "metadata" / "documentos.json"
CHUNKS_FILE = BASE_DIR / "corpus" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    ", ",
    " ",
    "",
]


def load_metadata():
    with open(METADATA_FILE) as f:
        return json.load(f)


def chunk_document(doc_meta):
    pdf_path = CORPUS_PDF / doc_meta["archivo"]
    if not pdf_path.exists():
        print(f"  ✗ File not found: {pdf_path}")
        return []

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    if not pages:
        print(f"  ✗ No pages extracted: {doc_meta['archivo']}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(pages)

    source_stem = Path(doc_meta["archivo"]).stem
    result = []
    for idx, doc in enumerate(chunks):
        text = doc.page_content.strip()
        if not text:
            continue

        meta = doc.metadata
        page = meta.get("page", meta.get("pages", ""))
        page_range = str(page) if page else ""

        result.append({
            "chunk_id": f"{source_stem}_{idx:04d}",
            "text": text,
            "metadata": {
                "category": doc_meta["categoria"],
                "document_title": doc_meta["titulo"],
                "author": doc_meta.get("autor", "Desconocido"),
                "institution": doc_meta.get("institucion", "Desconocida"),
                "year": doc_meta.get("anio"),
                "source_file": doc_meta["archivo"],
                "pages": page_range,
                "chunk_index": idx,
            },
        })
    return result


def run():
    print("=" * 60)
    print("  CHUNKING - Extracción y división de PDFs")
    print("=" * 60)

    all_docs = load_metadata()
    print(f"  Documentos en metadata: {len(all_docs)}")

    all_chunks = []
    for doc in all_docs:
        label = f"  {doc['archivo']:<55}"
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"{label} {len(chunks):>4} chunks")
        sys.stdout.flush()

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Archivo: {CHUNKS_FILE}")

    by_cat = {}
    for c in all_chunks:
        cat = c["metadata"]["category"]
        by_cat[cat] = by_cat.get(cat, 0) + 1

    print()
    for cat in sorted(by_cat):
        print(f"  {cat:<30} {by_cat[cat]:>4} chunks")
    print("=" * 60)


if __name__ == "__main__":
    run()
