#!/usr/bin/env python3
"""
RAG pipeline: recupera chunks relevantes de ChromaDB y genera respuesta con Ollama.
Uso interactivo: python src/rag/rag.py
"""

import sys, json, requests, time
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent.parent.parent
VECTORDB_DIR = BASE_DIR / "vectordb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "seguridad_mexico"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """Eres un asistente experto en seguridad pública en México.
Usa SOLO el contexto proporcionado para responder.

Contexto:
{context}

Pregunta: {question}

Respuesta (basada únicamente en el contexto proporcionado):"""


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def load_vectorstore():
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


def ask_ollama_stream(prompt, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    response.raise_for_status()

    full = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            chunk = data.get("response", "")
            full += chunk
            yield chunk
    return full


def query_once(question, k=5):
    print(f"  Loading vectorstore...", end=" ", flush=True)
    t0 = time.time()
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"  Retrieving chunks...", end=" ", flush=True)
    docs = retriever.invoke(question)
    print(f"done ({len(docs)} docs)")

    context = format_docs(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    print(f"\n  qwen2.5:7b en CPU (~0.5 tok/s), esperando...")
    t0 = time.time()
    print(f"\n  Respuesta:\n", end="", flush=True)
    for token in ask_ollama_stream(prompt):
        print(token, end="", flush=True)
    t1 = time.time()
    print(f"\n\n  Tiempo: {t1-t0:.0f}s")

    print(f"\n  Fuentes ({len(docs)}):")
    for i, doc in enumerate(docs):
        meta = doc.metadata
        src = meta.get("source_file", meta.get("source", "?"))
        pages = meta.get("pages", "?")
        cat = meta.get("category", "?")
        print(f"  [{i+1}] {src} (págs: {pages}, categoría: {cat})")


def run_cli():
    print(f"  Loading vectorstore...", end=" ", flush=True)
    t0 = time.time()
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    print(f"done ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 60)
    print("  RAG - Seguridad Pública México")
    print("  Escribe 'salir' para terminar")
    print("  (qwen2.5:7b en CPU ~0.5 tok/s)")
    print("=" * 60)

    last_docs = None

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit"):
            break
        if question.lower() == "fuentes" and last_docs:
            print("\n  Fuentes:")
            for i, doc in enumerate(last_docs):
                meta = doc.metadata
                src = meta.get("source_file", meta.get("source", "?"))
                pages = meta.get("pages", "?")
                cat = meta.get("category", "?")
                print(f"  [{i+1}] {src} (págs: {pages}, categoría: {cat})")
                print(f"      {doc.page_content[:100]}...")
            continue
        if question.lower() == "fuentes":
            print("  No hay resultados previos.")
            continue

        docs = retriever.invoke(question)
        last_docs = docs
        context = format_docs(docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        print(f"\n  qwen2.5:7b en CPU (~0.5 tok/s), esperando...")
        print(f"  Respuesta:\n", end="", flush=True)
        for token in ask_ollama_stream(prompt):
            print(token, end="", flush=True)
        print(f"\n  ---")
        print(f"  Fuentes: {len(docs)} chunks")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] not in ("-i", "--interactive"):
        query = " ".join(args)
        query_once(query)
    else:
        run_cli()
