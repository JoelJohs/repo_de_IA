#!/usr/bin/env python3
"""
Comparativa antes/después del RAG.
Ejecuta las 10 preguntas oficiales con dos modos:
  - before: RAG actual (Ollama qwen2.5:7b)
  - after:  RAG + LoRA (Qwen0.5B + adapter)

Genera tabla comparativa y guarda resultados en results/.
"""

import json, time, sys, os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from _format import (
    format_docs_with_attribution,
    load_manifesto,
    build_rag_prompt,
)

RESULTS_DIR = BASE_DIR / "results"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

OFFICIAL_QS = BASE_DIR / "src" / "evaluation" / "official_questions.json"

BEHAVIORS = {
    "cite_source": ["según", "de acuerdo con", "reporta", "documenta", "señala", "indica"],
    "neutral": ["diferentes perspectivas", "por un lado", "complementarias", "divergentes", "multifactorial"],
    "uncertainty": ["no detalla", "no contiene", "no es posible", "fuera del alcance", "insuficiente"],
    "socratic": ["antes de responder", "preguntarnos", "paso a paso", "qué evidencia", "revisemos"],
}


def load_vectorstore():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(BASE_DIR / "vectordb"),
        embedding_function=embeddings,
        collection_name="seguridad_mexico",
    )


def retrieve_context(retriever, question, k=5):
    docs = retriever.invoke(question)
    context = format_docs_with_attribution(docs)
    sources = [d.metadata.get("source_file", "?") for d in docs]
    return context, sources


def ask_ollama(prompt, stream=False):
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    resp = requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=120)
    resp.raise_for_status()
    if stream:
        full = ""
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("response", "")
                full += chunk
        return full
    return resp.json().get("response", "")


def prepare_prompt_official(question, context):
    manifesto = load_manifesto()
    manifesto_section = f"{manifesto}\n\n" if manifesto else ""
    return (
        f"{manifesto_section}"
        "Eres un asistente experto en seguridad pública en México. "
        "Responde basándote ÚNICAMENTE en el contexto proporcionado. "
        "Si el contexto no contiene la información, indícalo claramente.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n\n"
        "Respuesta:"
    )


def load_lora_adapter():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    ADAPTER_DIR = BASE_DIR / "src" / "finetuning" / "lora_adapter"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True,
    )

    adapter_path = ADAPTER_DIR
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        print("  WARNING: No LoRA adapter found, using base model")
        model = base
    model.eval()
    return tokenizer, model


def answer_with_lora(question, context, tokenizer, model):
    prompt = build_rag_prompt(context, question, include_manifesto=True)
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, temperature=0.1,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1].strip()
    return answer


def check_behavior(answer, behavior_keywords):
    answer_lower = answer.lower()
    hits = sum(1 for kw in behavior_keywords if kw in answer_lower)
    return hits, len(behavior_keywords)


def run_test(questions, mode, retriever, tokenizer=None, model=None):
    results = []
    times = []

    for q in questions:
        qid = q["id"]
        question = q["pregunta"]
        nivel = q["nivel"]

        print(f"  [{qid}] {question[:70]}...  ", end="", flush=True)
        t0 = time.time()

        context, sources = retrieve_context(retriever, question)
        if mode == "before":
            prompt = prepare_prompt_official(question, context)
            answer = ask_ollama(prompt, stream=True)
        else:
            answer = answer_with_lora(question, context, tokenizer, model)

        latency = time.time() - t0
        times.append(latency)

        # Behavior analysis
        behavior_scores = {}
        for bname, kws in BEHAVIORS.items():
            hits, total = check_behavior(answer, kws)
            behavior_scores[bname] = {"hits": hits, "total": total, "score": round(hits / max(total, 1), 2)}

        results.append({
            "id": qid,
            "nivel": nivel,
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency": round(latency, 1),
            "behavior_scores": behavior_scores,
        })

        print(f"{latency:.0f}s | {len(answer)} chars")
        sys.stdout.flush()

    return results


def print_table(results_before, results_after):
    print("\n" + "=" * 80)
    print("  COMPARATIVA: RAG Actual (Ollama 7B) vs RAG + LoRA (Qwen0.5B)")
    print("=" * 80)

    for rb, ra in zip(results_before, results_after):
        qid = rb["id"]
        nivel = rb["nivel"]
        print(f"\n  [{qid}] ({nivel}) {rb['question'][:60]}...")
        print(f"  {'─' * 70}")
        print(f"  BEFORE ({rb['latency']}s): {rb['answer'][:120]}...")

        b_scores = rb["behavior_scores"]
        a_scores = ra["behavior_scores"]
        score_line_b = " | ".join(f"{k}={v['score']}" for k, v in b_scores.items())
        score_line_a = " | ".join(f"{k}={v['score']}" for k, v in a_scores.items())
        print(f"  Scores:   {score_line_b}")
        print(f"  AFTER  ({ra['latency']}s): {ra['answer'][:120]}...")
        print(f"  Scores:   {score_line_a}")
        print(f"  Fuentes:  {len(rb['sources'])} docs")

    # Summary
    avg_lat_b = sum(r["latency"] for r in results_before) / len(results_before)
    avg_lat_a = sum(r["latency"] for r in results_after) / len(results_after)

    print("\n" + "=" * 80)
    print("  RESUMEN")
    print("=" * 80)
    print(f"  Latencia promedio BEFORE:  {avg_lat_b:.1f}s")
    print(f"  Latencia promedio AFTER:   {avg_lat_a:.1f}s")
    print(f"  Diferencia:                {avg_lat_b - avg_lat_a:.1f}s ({((avg_lat_a - avg_lat_b)/avg_lat_b*100):+.0f}%)")

    for behavior in BEHAVIORS:
        avg_b = sum(r["behavior_scores"][behavior]["score"] for r in results_before) / len(results_before)
        avg_a = sum(r["behavior_scores"][behavior]["score"] for r in results_after) / len(results_after)
        print(f"  {behavior:15s}: BEFORE={avg_b:.2f}  AFTER={avg_a:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after", "both"], default="both")
    parser.add_argument("--save", action="store_true", help="Guardar resultados en results/")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = json.load(open(OFFICIAL_QS))

    print("=" * 60)
    print("  COMPARATIVA RAG - Antes vs Después")
    print("=" * 60)

    print("  Loading vectorstore...")
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    results_before = None
    results_after = None

    if args.mode in ("before", "both"):
        print(f"\n  MODO: BEFORE (Ollama {OLLAMA_MODEL})\n")
        results_before = run_test(questions, "before", retriever)

    if args.mode in ("after", "both"):
        print(f"\n  MODO: AFTER (LoRA Qwen0.5B)\n")
        tokenizer, model = load_lora_adapter()
        results_after = run_test(questions, "after", retriever, tokenizer, model)

    if results_before and results_after:
        print_table(results_before, results_after)

    # Save results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_before:
            with open(RESULTS_DIR / f"before_{timestamp}.json", "w") as f:
                json.dump(results_before, f, indent=2, ensure_ascii=False)
            print(f"\n  Saved: results/before_{timestamp}.json")
        if results_after:
            with open(RESULTS_DIR / f"after_{timestamp}.json", "w") as f:
                json.dump(results_after, f, indent=2, ensure_ascii=False)
            print(f"  Saved: results/after_{timestamp}.json")


if __name__ == "__main__":
    main()
