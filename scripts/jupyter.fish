#!/usr/bin/env fish
# Inicia Jupyter Notebook en el navegador, con el venv unificado de la raiz
# del repo (Python 3.11 + Jupyter + TF + todo lo necesario para los 3
# proyectos). Navega a cualquier subcarpeta del repo (1-game, 2-CNN, 3-RNN).
#
# Uso:
#   ./scripts/jupyter.fish
#   ./scripts/jupyter.fish lab                  # usa JupyterLab
#   ./scripts/jupyter.fish notebook            # explicito (default)
#   ./scripts/jupyter.fish --port 8890         # cambia el puerto
#   ./scripts/jupyter.fish lab --port 9000     # lab + puerto custom
#
# Kernels disponibles (al abrir un notebook, elegi en la esquina superior derecha):
#   - python3 / IA (Python 3.11 + TF) : el venv unificado en IA/.venv
#     (tiene TF 2.16.1, fastapi, pygame, sklearn, etc. — todo lo
#     necesario para los 3 proyectos).
#
# Requisitos:
#   - pyenv con 3.11.9 instalado
#   - .venv unificado en la raiz del repo con jupyter + ipykernel

set script_dir (dirname (realpath (status filename)))
set repo_root (dirname $script_dir)
set python_bin "$repo_root/.venv/bin/python"

if not test -x $python_bin
    echo "[jupyter] No se encontro el venv unificado en $python_bin" >&2
    echo "[jupyter] Setup inicial:" >&2
    echo "    cd $repo_root" >&2
    echo "    pyenv local 3.11.9" >&2
    echo "    pyenv exec python -m venv .venv" >&2
    echo "    source .venv/bin/activate.fish" >&2
    echo "    pip install jupyter ipykernel matplotlib numpy" >&2
    echo "    # + dependencias de los proyectos:" >&2
    echo "    pip install -r 1-game/requirements.txt" >&2
    echo "    pip install -r 2-CNN/requirements.txt" >&2
    echo "    pip install -r 3-RNN/requirements.txt" >&2
    exit 1
end

# Verifica version de Python
set py_version ($python_bin --version 2>&1 | string replace -r 'Python ' '')
if not string match -q '3.11.*' $py_version
    echo "[jupyter] Aviso: el venv unificado usa Python $py_version, se esperaba 3.11.x" >&2
    read -l -P "Continuar de todas formas? [s/N] " confirm
    switch $confirm
        case s S si Si SI yes
            # continuar
        case '*'
            echo "[jupyter] Abortado. Recrear el venv con: cd $repo_root && pyenv exec python -m venv .venv" >&2
            exit 1
    end
end

# Verifica jupyter
if not $python_bin -c "import jupyter" 2>/dev/null
    echo "[jupyter] Falta jupyter en el venv." >&2
    read -l -P "Instalar ahora? [S/n] " confirm
    switch $confirm
        case n N no No NO
            echo "Abortado. Ejecuta: $python_bin -m pip install jupyter" >&2
            exit 1
    end
    $python_bin -m pip install jupyter
end

# Parsea modo (notebook|lab) y resto de args
set mode "notebook"
set extra_args
if test (count $argv) -ge 1
    switch $argv[1]
        case lab notebook
            set mode $argv[1]
            if test (count $argv) -ge 2
                set extra_args $argv[2..-1]
            end
        case '*'
            set extra_args $argv
    end
end

# Cambia al root del repo para que el file browser muestre todo
cd $repo_root

# PYTHONPATH para que los imports relativos de cada proyecto funcionen
set -x PYTHONPATH "$repo_root/1-game:$repo_root/2-CNN:$repo_root/3-RNN:$repo_root/4-RAG"

echo "[jupyter] Iniciando $mode con Python $py_version (venv unificado)..."
echo "[jupyter] Root: $repo_root"
echo "[jupyter] URL:   http://localhost:8888"
echo "[jupyter] Kernel: python3 (apunta al venv unificado)"
echo "[jupyter] PYTHONPATH: 1-game, 2-CNN, 3-RNN, 4-RAG"
echo

# ─── Imprimir tabla de contenidos de notebooks ───
echo "============================================================"
echo "  NOTA: Al abrir Jupyter, navega a notebooks/index.ipynb"
echo "        para ver el indice con todos los notebooks."
echo "============================================================"
echo
echo "  Proyecto   | Ruta"
echo "  ─────────────────────────────────────────────────────────"
echo "  1-game     | 1-game/docs/notebooks/"
echo "             |   explicacion_juego_pygame_mlp.ipynb"
echo "  2-CNN      | 2-CNN/"
echo "             |   CNN.ipynb"
echo "             |   CNNriesgo.ipynb"
echo "             | 2-CNN/docs/notebooks/"
echo "             |   explicacion_cnn_animales_paso_a_paso.ipynb"
echo "  3-RNN      | 3-RNN/notebooks/"
echo "             |   entrenamiento.ipynb"
echo "             | 3-RNN/docs/notebooks/"
echo "             |   explicacion_rnn_autocomplete_paso_a_paso.ipynb"
echo "  4-RAG      | 4-RAG/notebooks/"
echo "             |   rag_seguridad_mexico.ipynb"
echo
echo "  Total: 7 notebooks"
echo "============================================================"
echo

exec $python_bin -m jupyter $mode $extra_args
