#!/usr/bin/env python3
"""
Genera embeddings con all-MiniLM-L6-v2 y almacena en ChromaDB.
Carga chunks desde corpus/processed/chunks.jsonl, los embeddea
con sentence-transformers y los persiste en vectordb/.
"""

import json, sys, shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent.parent.parent
CHUNKS_FILE = BASE_DIR / "corpus" / "processed" / "chunks.jsonl"
VECTORDB_DIR = BASE_DIR / "vectordb"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "seguridad_mexico"


def load_chunks_as_documents():
    docs = []
    with open(CHUNKS_FILE) as f:
        for line in f:
            c = json.loads(line)
            meta = c["metadata"].copy()
            meta.pop("chunk_index", None)
            doc = Document(page_content=c["text"], metadata=meta, id=c["chunk_id"])
            docs.append(doc)
    return docs


def build_vectorstore(rebuild=False):
    if VECTORDB_DIR.exists() and not rebuild:
        print("  VectorDB already exists. Use --rebuild to reindex.")
        return

    print(f"  Loading chunks from {CHUNKS_FILE}")
    docs = load_chunks_as_documents()
    print(f"  Total chunks: {len(docs)}")

    print(f"  Loading model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    if VECTORDB_DIR.exists():
        shutil.rmtree(VECTORDB_DIR)
    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    print("  Embedding documents and building index...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTORDB_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"  Done. Collection '{COLLECTION_NAME}' with {vectordb._collection.count()} documents.")

    info = {
        "model": MODEL_NAME,
        "collection": COLLECTION_NAME,
        "total_chunks": len(docs),
        "dimension": 384,
    }
    with open(VECTORDB_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(VECTORDB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    build_vectorstore(rebuild=rebuild)
