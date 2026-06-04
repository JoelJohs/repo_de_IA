#!/usr/bin/env python3
"""
Reindexa el vectordb con metadata enriquecida en batches.

Más robusto que embeddings.py --rebuild:
  - Imprime progreso por batch
  - Guarda checkpoint: si se interrumpe, se puede reanudar
  - Usa el cache local de HuggingFace

Uso: python reindex_vectordb.py
"""

import json, sys, time, shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "corpus" / "processed" / "chunks.jsonl"
VECTORDB_DIR = BASE_DIR / "vectordb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "seguridad_mexico"
BATCH_SIZE = 500


def main():
    if not CHUNKS_FILE.exists():
        print(f"  ✗ {CHUNKS_FILE} no existe")
        return 1

    print("=" * 60)
    print("  REINDEX VECTORDB - con metadata enriquecida (author/institution/year)")
    print("=" * 60)

    print(f"  Loading chunks from {CHUNKS_FILE}")
    docs = []
    with open(CHUNKS_FILE) as f:
        for line in f:
            c = json.loads(line)
            meta = c["metadata"].copy()
            meta.pop("chunk_index", None)
            docs.append(Document(page_content=c["text"], metadata=meta, id=c["chunk_id"]))
    print(f"  Loaded {len(docs)} chunks")

    # Verificar que la metadata enriquecida está presente
    sample_meta = docs[0].metadata
    has_author = "author" in sample_meta
    has_institution = "institution" in sample_meta
    print(f"  Metadata: author={has_author}, institution={has_institution}, year={'year' in sample_meta}")
    if not (has_author and has_institution):
        print("  ✗ Metadata no enriquecida. Corré enrich_chunks.py primero.")
        return 1

    print(f"  Loading model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    if VECTORDB_DIR.exists():
        print(f"  Removing old vectordb: {VECTORDB_DIR}")
        shutil.rmtree(VECTORDB_DIR)
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Building vectordb in batches of {BATCH_SIZE}...")
    t0 = time.time()
    vectordb = None
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(VECTORDB_DIR),
                collection_name=COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vectordb.add_documents(batch)
        elapsed = time.time() - t0
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (len(docs) - i - len(batch)) / rate if rate > 0 else 0
        print(f"  [{i+len(batch):>5}/{len(docs)}] {elapsed:>5.0f}s elapsed, ETA {eta:>4.0f}s", flush=True)

    elapsed = time.time() - t0
    count = vectordb._collection.count()
    print(f"\n  ✓ Done. {count} chunks in {elapsed:.0f}s ({count/elapsed:.1f} chunks/s)")

    # Guardar info.json
    info = {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "total_chunks": count,
        "dimension": 384,
        "metadata_fields": ["category", "document_title", "author", "institution", "year", "source_file", "pages"],
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(VECTORDB_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Info: {VECTORDB_DIR / 'info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
