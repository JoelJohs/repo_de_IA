#!/usr/bin/env fish
# Inferencia con CNN entrenada para clasificacion de animales.
# Para entrenar: ./scripts/trains/train_cnn.fish
#
# Uso:
#   ./scripts/cnn.fish --predict modelo.keras imagen.jpg    # una imagen
#   ./scripts/cnn.fish --predict modelo.keras directorio/   # todo un directorio
#   ./scripts/cnn.fish -h / --help                          # esta ayuda

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname $script_dir)
set python_bin   "$repo_root/.venv/bin/python"
set project_root "$repo_root/2-CNN"
set config_path  "$project_root/config/default.yaml"

if not test -x $python_bin
    echo "[cnn] No se encontro el venv en $python_bin" >&2
    echo "[cnn] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

switch $argv[1]
    case --predict
        if test (count $argv) -lt 3
            echo "[cnn] Uso: ./scripts/cnn.fish --predict modelo.keras imagen.jpg" >&2
            echo "[cnn]       ./scripts/cnn.fish --predict modelo.keras directorio/" >&2
            exit 1
        end
        exec $python_bin "$project_root/predict.py" \
            --config $config_path --model $argv[2] --image $argv[3]

    case -h --help
        echo "Uso: (basename (status filename)) --predict MODELO IMAGEN"
        echo
        echo "  --predict MODELO IMAGEN   inferencia sobre una imagen o directorio"
        echo
        echo "Ejemplos:"
        echo "  ./scripts/cnn.fish --predict 2-CNN/models/best_model.keras imagen.jpg"
        echo "  ./scripts/cnn.fish --predict 2-CNN/models/best_model.keras 2-CNN/test/"
        echo
        echo "Para ENTRENAR el modelo:"
        echo "  ./scripts/trains/train_cnn.fish"
        exit 0

    case '*'
        echo "[cnn] Uso: ./scripts/cnn.fish --predict MODELO IMAGEN" >&2
        echo "[cnn]" >&2
        echo "[cnn] Para ENTRENAR el modelo:" >&2
        echo "[cnn]   ./scripts/trains/train_cnn.fish" >&2
        exit 1
end
