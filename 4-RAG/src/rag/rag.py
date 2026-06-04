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
sys.path.insert(0, str(BASE_DIR / "src"))
from _format import (
    format_docs_with_attribution,
    load_manifesto,
    build_rag_prompt,
)

VECTORDB_DIR = BASE_DIR / "vectordb"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "seguridad_mexico"
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE_OLLAMA = """{manifesto_section}Eres un asistente experto en seguridad pública en México.
Responde SOLO con la información del contexto proporcionado.
Cuando el contexto incluya datos cuantitativos, cita la institución y el título.
Si hay disenso entre autores, presenta las múltiples perspectivas.

Contexto:
{context}

Pregunta: {question}

Respuesta (basada únicamente en el contexto proporcionado):"""

# Frases introductorias comunes que Ollama a veces añade antes de responder
SKIP_PREFIXES = [
    "aquí está", "aquí tienes", "claro", "por supuesto",
    "basado", "de acuerdo", "respuesta:", "answer:",
]


def format_docs(docs):
    return format_docs_with_attribution(docs)


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


def ask_ollama_stream(prompt, model):
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


def query_once(question, k=5, model=OLLAMA_MODEL):
    print(f"  Cargando vectorstore...", end=" ", flush=True)
    t0 = time.time()
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    print(f"hecho ({time.time()-t0:.1f}s)")
    print(f"  Recuperando chunks...", end=" ", flush=True)
    docs = retriever.invoke(question)
    print(f"hecho ({len(docs)} chunks)")

    show_sources(docs, "Contexto")

    context = format_docs(docs)
    manifesto = load_manifesto()
    manifesto_section = f"{manifesto}\n\n" if manifesto else ""
    prompt = PROMPT_TEMPLATE_OLLAMA.format(
        manifesto_section=manifesto_section,
        context=context,
        question=question,
    )

    print(f"\n  Generando respuesta ({model})...\n", flush=True)
    t0 = time.time()
    for token in ask_ollama_stream(prompt, model):
        print(token, end="", flush=True)
    t1 = time.time()
    print(f"\n\n  Tiempo: {t1-t0:.0f}s")


def show_sources(docs, title="Fuentes"):
    print(f"\n  {title} ({len(docs)} chunks):")
    for i, doc in enumerate(docs):
        meta = doc.metadata
        src = meta.get("source_file", meta.get("source", "?"))
        pages = meta.get("pages", "?")
        cat = meta.get("category", "?")
        print(f"  [{i+1}] {src} (págs: {pages}, categoría: {cat})")
        print(f"      {doc.page_content[:120]}...")


def strip_prefix(text):
    lower = text.lower().strip()
    for prefix in SKIP_PREFIXES:
        if lower.startswith(prefix):
            cutoff = text[len(prefix):]
            # si el prefijo va seguido de dos puntos o salto, limpiar
            return cutoff.lstrip(":,. -\n\r")
    return text


def run_cli(model=OLLAMA_MODEL):
    print(f"  Cargando vectorstore...", end=" ", flush=True)
    t0 = time.time()
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    print(f"hecho ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 60)
    print("  RAG - Seguridad Pública México")
    print("  Escribe 'salir' para terminar")
    print(f"  Modelo: {model} (CPU ~0.5 tok/s)")
    print("  Comandos: /fuentes  /modelo NOMBRE  /salir")
    print("=" * 60)

    last_docs = None

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saliendo...")
            break

        if not question:
            continue
        if question.lower() in ("salir", "exit", "quit", "/salir"):
            break

        # comandos especiales
        if question.lower() == "/fuentes":
            if last_docs:
                show_sources(last_docs)
            else:
                print("  No hay consulta previa.")
            continue
        if question.lower().startswith("/modelo"):
            parts = question.split(maxsplit=1)
            if len(parts) > 1:
                model = parts[1].strip()
                print(f"  Modelo cambiado a: {model}")
            else:
                print(f"  Modelo actual: {model}")
            continue

        # recuperar chunks
        print(f"  Buscando chunks relevantes...", end=" ", flush=True)
        docs = retriever.invoke(question)
        last_docs = docs
        print(f"({len(docs)} chunks encontrados)")
        show_sources(docs, "Contexto recuperado")

        # generar respuesta
        context = format_docs(docs)
        manifesto = load_manifesto()
        manifesto_section = f"{manifesto}\n\n" if manifesto else ""
        prompt = PROMPT_TEMPLATE_OLLAMA.format(
            manifesto_section=manifesto_section,
            context=context,
            question=question,
        )

        print(f"\n  Generando respuesta ({model})...\n", flush=True)
        try:
            accumulated = ""
            for token in ask_ollama_stream(prompt, model):
                accumulated += token
                print(token, end="", flush=True)
            # limpiar prefijo si aparece
            cleaned = strip_prefix(accumulated)
            if cleaned != accumulated:
                # reimprimir sin prefijo
                print(f"\r  {cleaned}", end="", flush=True)
        except KeyboardInterrupt:
            print("\n  Generación interrumpida.")
        except Exception as e:
            print(f"\n  Error: {e}")

        print(f"\n\n  ---")


if __name__ == "__main__":
    args = sys.argv[1:]
    model = OLLAMA_MODEL

    # parsear --model delante
    filtered = []
    i = 0
    while i < len(args):
        if args[i] in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] in ("-i", "--interactive"):
            i += 1
        else:
            filtered.append(args[i])
            i += 1
    args = filtered

    if args:
        query = " ".join(args)
        query_once(query, model=model)
    else:
        run_cli(model=model)
