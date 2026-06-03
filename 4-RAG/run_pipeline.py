#!/usr/bin/env python3
"""
Orquestador del pipeline RAG:
  1. chunking    — extraer texto y dividir en chunks
  2. embeddings  — generar embeddings e indexar en ChromaDB
  3. rag         — (opcional) lanzar RAG interactivo
  4. evaluate    — (opcional) evaluación de recuperación
  5. api         — (opcional) API FastAPI
  6. dataset     — (opcional) generar dataset para fine-tuning
  7. finetune    — (opcional) entrenar LoRA
  8. integrate   — (opcional) RAG con LoRA integrado

Uso:
  python run_pipeline.py                    # chunking + embeddings
  python run_pipeline.py --rag              # + modo interactivo
  python run_pipeline.py --evaluate         # + evaluación
  python run_pipeline.py --api              # + API
  python run_pipeline.py --generate-dataset # + generar dataset
  python run_pipeline.py --rebuild          # forzar reindexación
  python run_pipeline.py --skip-chunking    # solo embeddings
  python run_pipeline.py --skip-embeddings  # solo chunking
"""

import sys, subprocess, argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
VENV_PYTHON = BASE_DIR.parent / ".venv" / "bin" / "python"


def run_script(script_path, extra_args=None):
    cmd = [str(VENV_PYTHON), str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[pipeline] Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"[pipeline] ERROR: script exited with code {result.returncode}")
        sys.exit(result.returncode)
    print()


def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG - Seguridad Pública México")
    parser.add_argument("--rag", action="store_true", help="Lanzar RAG interactivo al final")
    parser.add_argument("--rebuild", action="store_true", help="Forzar reindexación de vectordb")
    parser.add_argument("--skip-chunking", action="store_true", help="Saltar paso de chunking")
    parser.add_argument("--skip-embeddings", action="store_true", help="Saltar paso de embeddings")
    parser.add_argument("--api", action="store_true", help="Iniciar API FastAPI")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para la API")
    parser.add_argument("--evaluate", action="store_true", help="Ejecutar evaluación")
    parser.add_argument("--generate-dataset", action="store_true", help="Generar dataset QA técnico desde chunks")
    parser.add_argument("--generate-behavioral", action="store_true", help="Generar dataset comportamental (300 ejemplos)")
    parser.add_argument("--finetune", action="store_true", help="Entrenar LoRA sobre Qwen2.5-0.5B")
    parser.add_argument("--integrate", action="store_true", help="RAG con LoRA integrado (modo interactivo)")
    parser.add_argument("--compare", nargs="?", const="both", choices=["before", "after", "both"],
                        help="Comparativa antes/después del RAG")
    parser.add_argument("--official", action="store_true",
                        help="Responder las 10 preguntas oficiales con RAG actual")
    args = parser.parse_args()

    print("=" * 60)
    print("  PIPELINE RAG - Seguridad Pública en México")
    print("=" * 60)

    if not args.skip_chunking:
        run_script(BASE_DIR / "src" / "ingest" / "chunking.py")

    if not args.skip_embeddings:
        extra = ["--rebuild"] if args.rebuild else []
        run_script(BASE_DIR / "src" / "embeddings" / "embeddings.py", extra)

    if args.rag:
        run_script(BASE_DIR / "src" / "rag" / "rag.py", ["-i"])

    if args.evaluate:
        run_script(BASE_DIR / "src" / "evaluation" / "evaluate.py")

    if args.api:
        print(f"[pipeline] Starting API on port {args.port}...")
        subprocess.run(
            [str(VENV_PYTHON), "-m", "uvicorn", "src.api.main:app",
             "--host", "0.0.0.0", "--port", str(args.port)],
            cwd=BASE_DIR,
        )

    if args.generate_dataset:
        run_script(BASE_DIR / "src" / "finetuning" / "generate_dataset.py")

    if args.generate_behavioral:
        run_script(BASE_DIR / "datasets" / "behavioral_dataset.py")

    if args.finetune:
        run_script(BASE_DIR / "src" / "finetuning" / "finetune.py")

    if args.integrate:
        run_script(BASE_DIR / "src" / "finetuning" / "integrate.py", ["-i"])

    if args.compare:
        run_script(BASE_DIR / "src" / "evaluation" / "compare.py", ["--mode", args.compare, "--save"])

    if args.official:
        run_script(BASE_DIR / "src" / "evaluation" / "compare.py", ["--mode", "before", "--save"])

    print("[pipeline] Done.")


if __name__ == "__main__":
    main()
