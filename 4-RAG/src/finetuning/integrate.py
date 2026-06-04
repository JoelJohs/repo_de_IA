#!/usr/bin/env python3
"""
Integra el adaptador LoRA entrenado en el pipeline RAG.
Carga el modelo base Qwen2.5-0.5B-Instruct + el adaptador LoRA,
y lo usa como generador en lugar de la llamada directa a Ollama.

Uso:  python src/finetuning/integrate.py --query "¿Cuántos homicidios hubo en 2024?"
"""

import sys, json, time, argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from _format import (
    format_docs_with_attribution,
    load_manifesto,
    build_rag_prompt,
)

ADAPTER_DIR = BASE_DIR / "src" / "finetuning" / "lora_adapter"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"


class LoRARAG:
    def __init__(self, adapter_path=None):
        self.adapter_path = adapter_path or ADAPTER_DIR
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"  Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, local_files_only=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=False,
        )

        if self.adapter_path.exists():
            print(f"  Loading LoRA adapter from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base, str(self.adapter_path))
        else:
            print(f"  No adapter found at {self.adapter_path}, using base model")
            self.model = base

        self.model.eval()

    def generate(self, prompt, max_new_tokens=256, temperature=0.1):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_retriever():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    VECTORDB_DIR = BASE_DIR / "vectordb"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    COLLECTION_NAME = "seguridad_mexico"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, help="Pregunta")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo")
    parser.add_argument("--k", type=int, default=5, help="Número de chunks a recuperar")
    parser.add_argument("--use-ollama", action="store_true", help="Usar Ollama en lugar de LoRA")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG + LoRA - Seguridad Pública México")
    print("=" * 60)

    print("  Loading vectorstore...")
    t0 = time.time()
    vectordb = load_retriever()
    print(f"  done ({time.time()-t0:.1f}s)")

    if not args.use_ollama:
        lora_rag = LoRARAG()
    else:
        lora_rag = None

    def query(question, k=args.k):
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        context = format_docs_with_attribution(docs)

        if args.use_ollama:
            import requests
            manifesto = load_manifesto()
            manifesto_section = f"{manifesto}\n\n" if manifesto else ""
            prompt = (
                f"{manifesto_section}"
                "Eres un asistente experto en seguridad pública en México. "
                "Responde ÚNICAMENTE basado en el contexto.\n\n"
                f"Contexto:\n{context}\n\n"
                f"Pregunta: {question}\n\n"
                "Respuesta:"
            )
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": 0.1, "num_predict": 512},
            }
            resp = requests.post(OLLAMA_URL, json=payload, stream=True)
            resp.raise_for_status()
            print("\n  Respuesta:\n", end="", flush=True)
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    print(token, end="", flush=True)
            print()
        else:
            prompt = build_rag_prompt(context, question, include_manifesto=True)
            answer = lora_rag.generate(prompt)
            # Extract the assistant response part
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()
            print(f"\n  Respuesta:\n  {answer}")

        print(f"\n  Fuentes ({len(docs)}):")
        for i, d in enumerate(docs):
            src = d.metadata.get("source_file", "?")
            pages = d.metadata.get("pages", "?")
            cat = d.metadata.get("category", "?")
            inst = d.metadata.get("institution", "?")
            author = d.metadata.get("author", "?")
            year = d.metadata.get("year", "?")
            print(f"  [{i+1}] {inst} / {author} — {src} ({year}, pág. {pages}, cat: {cat})")

    if args.query:
        query(args.query)
    elif args.interactive:
        print("\n  Escribe 'salir' para terminar")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("salir", "exit", "quit"):
                break
            query(q)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
