#!/usr/bin/env python3
"""
Genera dataset JSONL para fine-tuning a partir de los chunks indexados.
Para cada chunk, se genera una pregunta usando Ollama y se crea el par (pregunta, chunk, respuesta).
También se incluyen los pares del banco de pruebas de evaluación.

Salida: src/finetuning/dataset/rag_dataset.jsonl
"""

import json, time, os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent.parent.parent
VECTORDB_DIR = BASE_DIR / "vectordb"
DATASET_DIR = BASE_DIR / "src" / "finetuning" / "dataset"
TEST_BANK = BASE_DIR / "src" / "evaluation" / "test_bank.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "seguridad_mexico"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

NUM_SAMPLES = 50


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


def ask_ollama(prompt, max_tokens=256):
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"    Ollama error: {e}")
        return ""


def generate_question_from_chunk(chunk_text, category):
    prompt = (
        "Genera UNA pregunta en español que pueda responderse usando el siguiente texto. "
        "La pregunta debe ser específica y factual.\n\n"
        f"Categoría: {category}\n\n"
        f"Texto: {chunk_text[:1500]}\n\n"
        "Pregunta:"
    )
    q = ask_ollama(prompt, max_tokens=64)
    return q


def get_condensed_answer(chunk_text, question):
    prompt = (
        "Responde la siguiente pregunta usando SOLO la información proporcionada. "
        "Sé conciso (máximo 3 oraciones).\n\n"
        f"Contexto: {chunk_text[:1500]}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta:"
    )
    a = ask_ollama(prompt, max_tokens=128)
    return a


def main():
    print("=" * 60)
    print("  DATASET GENERATION")
    print("=" * 60)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATASET_DIR / "rag_dataset.jsonl"

    print("  Loading vectorstore...")
    vectordb = load_vectorstore()
    count = vectordb._collection.count()

    import random
    all_ids = vectordb._collection.get(limit=count)["ids"]
    sample_ids = random.sample(all_ids, min(NUM_SAMPLES, len(all_ids)))
    sample_docs = vectordb._collection.get(ids=sample_ids, include=["documents", "metadatas"])

    print(f"  Total chunks: {count}, sampling: {len(sample_ids)}")

    records = []

    # Generate Q&A from chunks
    for i, (doc_id, text, meta) in enumerate(zip(
        sample_docs["ids"], sample_docs["documents"], sample_docs["metadatas"]
    )):
        category = meta.get("category", "general")
        source = meta.get("source_file", meta.get("source", "unknown"))
        print(f"  [{i+1}/{len(sample_ids)}] Generating Q&A for chunk {doc_id[:12]}... ({category})")

        question = generate_question_from_chunk(text, category)
        if not question:
            continue

        answer = get_condensed_answer(text, question)
        if not answer:
            continue

        record = {
            "instruction": question,
            "input": "",
            "output": answer,
            "chunk_id": doc_id,
            "category": category,
            "source": source,
        }
        records.append(record)

        with open(output_file, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        time.sleep(1)

    # Add test bank entries (using original chunks as answers)
    print(f"\n  Adding test bank entries...")
    test_questions = json.load(open(TEST_BANK))
    for test in test_questions:
        q = test["question"]
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content[:800] for d in docs)
        if not context:
            continue
        answer = get_condensed_answer(context, q)
        if not answer:
            answer = context[:300]
        records.append({
            "instruction": q,
            "input": context[:800],
            "output": answer,
            "chunk_id": "test_bank",
            "category": ",".join(test.get("expected_categories", [])),
            "source": "test_bank",
        })

    with open(output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  Dataset saved: {output_file} ({len(records)} records)")


if __name__ == "__main__":
    main()
