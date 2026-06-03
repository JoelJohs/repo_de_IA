#!/usr/bin/env fish
# Setup inicial del repositorio.
# Crea el venv, instala dependencias de los 4 proyectos, verifica requisitos.
#
# Uso:
#   ./scripts/setup.fish                # setup completo
#   ./scripts/setup.fish --quick        # solo verificar, no instalar
#   ./scripts/setup.fish --only RAG     # solo dependencias de 4-RAG

set script_dir (dirname (realpath (status filename)))
set repo_root  (dirname $script_dir)
set python_bin "$repo_root/.venv/bin/python"

set quick false
set only ""

# --- parseo de flags ---------------------------------------------------------
while test (count $argv) -gt 0
    switch $argv[1]
        case --quick
            set quick true
        case --only
            set only $argv[2]
            set argv $argv[2..-1]
        case -h --help
            echo "Uso: (basename (status filename)) [--quick] [--only PROYECTO]"
            echo
            echo "  --quick          solo verificar, saltar instalacion"
            echo "  --only PROYECTO  instalar solo un proyecto (game|CNN|RNN|RAG)"
            exit 0
        case '*'
            echo "[setup] Argumento desconocido: $argv[1]" >&2
            exit 1
    end
    set argv $argv[2..-1]
end

echo "============================================================"
echo "  SETUP - IA"
echo "  Root: $repo_root"
echo "============================================================"
echo

# --- 1) Python executable ----------------------------------------------------
echo "[setup] [1/5] Buscando Python..."
# Intentar pyenv primero (el proyecto usa .python-version → 3.11.9)
if command -v pyenv >/dev/null
    set pyenv_root (pyenv root 2>/dev/null)
    set pyenv_ver (cat "$repo_root/.python-version" 2>/dev/null)
    if test -n "$pyenv_ver"; and test -f "$pyenv_root/versions/$pyenv_ver/bin/python"
        set python_exec "$pyenv_root/versions/$pyenv_ver/bin/python"
        echo "[setup]       pyenv: $pyenv_ver"
    end
end

# Fallback: python3 del PATH
if not set -q python_exec
    if command -v python3 >/dev/null
        set python_exec (which python3)
    else
        echo "[setup] ERROR: python3 no encontrado." >&2
        exit 1
    end
end

set py_ver ($python_exec --version 2>&1 | string replace -r 'Python ' '')
echo "[setup]       Python: $py_ver ($python_exec)"

# --- 2) venv -----------------------------------------------------------------
echo "[setup] [2/5] Verificando venv..."
if not test -f $python_bin
    echo "[setup] Creando .venv con $python_exec..."
    $python_exec -m venv "$repo_root/.venv"
    if test $status -ne 0
        echo "[setup] ERROR: fallo al crear el venv" >&2
        exit 1
    end
    echo "[setup]       .venv creado"

    # Verificar que la version del venv coincida
    set venv_ver ($python_bin --version 2>&1 | string replace -r 'Python ' '')
    echo "[setup]       Version: $venv_ver"
else
    echo "[setup]       .venv existe ($python_bin)"
end

# --- 3) pip ------------------------------------------------------------------
echo "[setup] [3/5] Actualizando pip..."
$python_bin -m pip install --upgrade pip -q
echo "[setup]       pip actualizado"

# --- 4) dependencias ---------------------------------------------------------
echo "[setup] [4/5] Instalando dependencias..."

# Core (siempre)
set deps_installed false

# 1-game
if test -z "$only"; or test "$only" = "game"
    set req "$repo_root/1-game/requirements.txt"
    if test -f $req
        echo "[setup]       1-game..."
        $python_bin -m pip install -r $req -q
        set deps_installed true
    end
end

# 2-CNN
if test -z "$only"; or test "$only" = "CNN"
    set req "$repo_root/2-CNN/requirements.txt"
    if test -f $req
        echo "[setup]       2-CNN..."
        $python_bin -m pip install -r $req -q
        set deps_installed true
    end
end

# 3-RNN
if test -z "$only"; or test "$only" = "RNN"
    set req "$repo_root/3-RNN/requirements.txt"
    if test -f $req
        echo "[setup]       3-RNN..."
        $python_bin -m pip install -r $req -q
        set deps_installed true
    end
end

# 4-RAG
if test -z "$only"; or test "$only" = "RAG"
    set req "$repo_root/4-RAG/requirements.txt"
    if test -f $req
        echo "[setup]       4-RAG..."
        $python_bin -m pip install -r $req -q
        set deps_installed true
    end
end

# Jupyter + extras (si no se especifico --only)
if test -z "$only"
    $python_bin -m pip install jupyter ipykernel -q
end

echo "[setup]       dependencias instaladas"

# --- 5) verificaciones -------------------------------------------------------
echo "[setup] [5/5] Verificando paquetes base..."

function check_import
    set pkg $argv[1]
    set label $argv[2]
    $python_bin -c "import $pkg" 2>/dev/null
    if test $status -eq 0
        echo "[setup]       [OK] $label"
        return 0
    else
        echo "[setup]       [--] $label (no instalado)"
        return 1
    end
end

check_import pygame         "pygame"
check_import sklearn        "scikit-learn"
check_import tensorflow     "tensorflow"
check_import langchain_huggingface "langchain"
check_import chromadb       "chromadb"
check_import torch          "torch"
check_import fastapi        "fastapi"
check_import sentence_transformers "sentence-transformers"

echo
echo "[setup] Listo."
echo "[setup] Activar: source .venv/bin/activate.fish"
