#!/usr/bin/env python3
"""
Evaluation module for RAG pipeline.
Carga banco de pruebas, ejecuta retrieval y generación,
mide precisión, recall, latencia y faithfulness.
"""

import json, time, re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent.parent.parent
VECTORDB_DIR = BASE_DIR / "vectordb"
TEST_BANK = BASE_DIR / "src" / "evaluation" / "test_bank.json"
RESULTS_FILE = BASE_DIR / "src" / "evaluation" / "results.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "seguridad_mexico"
K = 5


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


def load_test_bank():
    with open(TEST_BANK) as f:
        return json.load(f)


def precision_at_k(retrieved_docs, expected_sources, k=K):
    if not expected_sources:
        return None
    relevant = sum(
        1 for d in retrieved_docs[:k]
        if any(src in d.metadata.get("source_file", "") for src in expected_sources)
    )
    return relevant / k


def recall_at_k(retrieved_docs, expected_sources, k=K):
    if not expected_sources:
        return None
    retrieved_files = [d.metadata.get("source_file", "") for d in retrieved_docs[:k]]
    found = sum(1 for src in expected_sources if any(src in rf for rf in retrieved_files))
    return found / len(expected_sources)


def category_match(retrieved_docs, expected_categories, k=K):
    retrieved_cats = set(d.metadata.get("category", "") for d in retrieved_docs[:k])
    expected = set(expected_categories)
    if not expected:
        return None
    return len(retrieved_cats & expected) / len(expected)


def keyword_match(answer, keywords):
    if not keywords:
        return None
    answer_lower = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matches / len(keywords)


def faithfulness_score(answer, context_texts):
    if not answer or not context_texts:
        return 0.0
    combined = " ".join(context_texts).lower()
    answer_sentences = re.split(r'[.!?]+', answer.lower())
    supported = 0
    total = 0
    for sent in answer_sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        total += 1
        words = set(sent.split())
        overlap = sum(1 for w in words if len(w) > 3 and w in combined)
        if overlap / max(len(words), 1) > 0.3:
            supported += 1
    return supported / max(total, 1)


def run_retrieval_eval(vectordb):
    tests = load_test_bank()
    retriever = vectordb.as_retriever(search_kwargs={"k": K})
    results = []
    retrieval_times = []
    generation_times = []

    print(f"  Running {len(tests)} retrieval tests...\n")
    for i, test in enumerate(tests):
        question = test["question"]
        q_short = question[:60] + ("..." if len(question) > 60 else "")

        t0 = time.time()
        docs = retriever.invoke(question)
        t1 = time.time()
        retrieval_time = t1 - t0
        retrieval_times.append(retrieval_time)

        p = precision_at_k(docs, test.get("expected_sources", []))
        r = recall_at_k(docs, test.get("expected_sources", []))
        cm = category_match(docs, test.get("expected_categories", []))

        row = {
            "question": question,
            "retrieval_time": round(retrieval_time, 2),
            "precision": round(p, 2) if p is not None else "N/A",
            "recall": round(r, 2) if r is not None else "N/A",
            "category_match": round(cm, 2) if cm is not None else "N/A",
            "retrieved_categories": list(set(d.metadata.get("category", "") for d in docs[:K])),
            "retrieved_sources": [d.metadata.get("source_file", "") for d in docs[:K]],
        }

        exp_cats = ", ".join(test.get("expected_categories", []))
        exp_str = ", ".join(test.get("expected_sources", [])) or "(any)"
        print(f"  [{i+1:02d}] {q_short}")
        print(f"       Expected: {exp_cats} | P@{K}={row['precision']} R@{K}={row['recall']} | {retrieval_time:.1f}s")

        results.append(row)

    # Summary
    precisions = [r["precision"] for r in results if r["precision"] != "N/A"]
    recalls = [r["recall"] for r in results if r["recall"] != "N/A"]
    cat_matches = [r["category_match"] for r in results if r["category_match"] != "N/A"]

    summary = {
        "total_tests": len(tests),
        "avg_precision_at_5": round(sum(precisions) / len(precisions), 2) if precisions else "N/A",
        "avg_recall_at_5": round(sum(recalls) / len(recalls), 2) if recalls else "N/A",
        "avg_category_match": round(sum(cat_matches) / len(cat_matches), 2) if cat_matches else "N/A",
        "avg_retrieval_time": round(sum(retrieval_times) / len(retrieval_times), 2),
        "max_retrieval_time": round(max(retrieval_times), 2),
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total tests:       {summary['total_tests']}")
    print(f"  Precision@{K}:       {summary['avg_precision_at_5']}")
    print(f"  Recall@{K}:          {summary['avg_recall_at_5']}")
    print(f"  Category match:    {summary['avg_category_match']}")
    print(f"  Avg retrieval:     {summary['avg_retrieval_time']}s")
    print(f"  Max retrieval:     {summary['max_retrieval_time']}s")
    print(f"  Results saved to:  {RESULTS_FILE}")


def run_full_eval(generate=False):
    print("=" * 60)
    print("  EVALUATION - RAG Seguridad Pública México")
    print("=" * 60)

    t0 = time.time()
    vectordb = load_vectorstore()
    print(f"  Vectorstore loaded ({time.time()-t0:.1f}s)\n")

    run_retrieval_eval(vectordb)

    if generate:
        print("\n  (Full generation eval requires Ollama running)")
        run_generation_eval(vectordb)


if __name__ == "__main__":
    import sys
    generate = "--generate" in sys.argv
    run_full_eval(generate=generate)
