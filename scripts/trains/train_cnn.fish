#!/usr/bin/env fish
# Entrena la CNN de 2-CNN para clasificacion de animales.
# Delegates a 2-CNN/scripts/train.fish.
#
# Uso:
#   ./scripts/trains/train_cnn.fish                              # config default
#   ./scripts/trains/train_cnn.fish ruta/a/config.yaml           # config alternativa

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname (dirname $script_dir))
set python_bin   "$repo_root/.venv/bin/python"
set project_root "$repo_root/2-CNN"
set config_path  "$project_root/config/default.yaml"

if not test -x $python_bin
    echo "[train_cnn] No se encontro el venv en $python_bin" >&2
    echo "[train_cnn] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

if not test -d "$project_root/dataset"
    echo "[train_cnn] No se encontro el dataset en $project_root/dataset" >&2
    echo "[train_cnn] Construyelo primero:" >&2
    echo "[train_cnn]   python -m 2-CNN.src.data.build_processed_dataset --config $config_path" >&2
    exit 1
end

if test (count $argv) -ge 2
    echo "Uso: (basename (status filename)) [config]" >&2
    exit 1
end

if test (count $argv) -ge 1
    set config_path $argv[1]
end

exec $python_bin "$project_root/train.py" --config $config_path
