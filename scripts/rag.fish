#!/usr/bin/env fish
# Orquesta el pipeline RAG (4-RAG) sobre seguridad publica en Mexico.
# Es un thin wrapper sobre run_pipeline.py.
#
# Uso:
#   ./scripts/rag.fish [OPCIONES]
#
#   (sin args)   → chat interactivo directo (sin pipeline)
#   --rebuild    → chunking + embeddings completos
#   --evaluate   → metricas de recuperacion
#   --api        → FastAPI :8000
#   --official   → 10 preguntas oficiales
#   --finetune   → LoRA Qwen0.5B
#   --model NAME → modelo Ollama alternativo
#   --help       → todos los flags

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname $script_dir)
set python_bin   "$repo_root/.venv/bin/python"
set pipeline     "$repo_root/4-RAG/run_pipeline.py"
set rag_script   "$repo_root/4-RAG/src/rag/rag.py"

if not test -x $python_bin
    echo "[rag] No se encontro el venv en $python_bin" >&2
    echo "[rag] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

# Sin argumentos → chat interactivo directo (evita pipeline y embeddings)
if test (count $argv) -eq 0
    if not test -f $rag_script
        echo "[rag] No se encontro $rag_script" >&2
        exit 1
    end
    echo "============================================================"
    echo "  RAG - Seguridad Publica Mexico (chat directo)"
    echo "============================================================"
    echo
    exec $python_bin $rag_script -i
end

# Con argumentos → pipeline
if not test -f $pipeline
    echo "[rag] No se encontro $pipeline" >&2
    exit 1
end

echo "============================================================"
echo "  RAG - Seguridad Publica Mexico"
echo "============================================================"
echo

exec $python_bin $pipeline $argv
