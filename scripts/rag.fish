#!/usr/bin/env fish
# Orquesta el pipeline RAG (4-RAG) sobre seguridad publica en Mexico.
# Es un thin wrapper sobre run_pipeline.py.
#
# Uso:
#   ./scripts/rag.fish                          # chunking + embeddings
#   ./scripts/rag.fish --rag                    # RAG interactivo
#   ./scripts/rag.fish --evaluate              # metricas de recuperacion
#   ./scripts/rag.fish --api                   # FastAPI :8000
#   ./scripts/rag.fish --generate-behavioral   # 300 ejemplos
#   ./scripts/rag.fish --finetune              # LoRA Qwen0.5B
#   ./scripts/rag.fish --integrate             # RAG + LoRA
#   ./scripts/rag.fish --compare both --save   # comparativa
#   ./scripts/rag.fish --official              # 10 preguntas oficiales
#   ./scripts/rag.fish --help                  # todos los flags

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname $script_dir)
set python_bin   "$repo_root/.venv/bin/python"
set pipeline     "$repo_root/4-RAG/run_pipeline.py"

if not test -x $python_bin
    echo "[rag] No se encontro el venv en $python_bin" >&2
    echo "[rag] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

if not test -f $pipeline
    echo "[rag] No se encontro $pipeline" >&2
    exit 1
end

echo "============================================================"
echo "  RAG - Seguridad Publica Mexico"
echo "============================================================"
echo

exec $python_bin $pipeline $argv
