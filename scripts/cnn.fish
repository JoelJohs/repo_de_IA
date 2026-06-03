#!/usr/bin/env fish
# Inferencia con CNN entrenada para clasificacion de animales.
# Sin args: corre sobre todo 2-CNN/test/ con el mejor modelo guardado.
# Para entrenar: ./scripts/trains/train_cnn.fish
#
# Uso:
#   ./scripts/cnn.fish                                    # test/ completo
#   ./scripts/cnn.fish modelo.keras                       # modelo custom + test/
#   ./scripts/cnn.fish modelo.keras imagen.jpg            # una imagen
#   ./scripts/cnn.fish modelo.keras directorio/           # directorio custom
#   ./scripts/cnn.fish modelo.keras imagen.jpg config.yaml
#   ./scripts/cnn.fish -h / --help                        # esta ayuda

set script_dir     (dirname (realpath (status filename)))
set repo_root      (dirname $script_dir)
set python_bin     "$repo_root/.venv/bin/python"
set project_root   "$repo_root/2-CNN"
set config_path    "$project_root/config/default.yaml"
set default_model  "$project_root/config/models/best_model.keras"
set default_images "$project_root/test"

if not test -x $python_bin
    echo "[cnn] No se encontro el venv en $python_bin" >&2
    echo "[cnn] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

if test "$argv[1]" = "-h"; or test "$argv[1]" = "--help"
    echo "Uso: (basename (status filename)) [modelo] [imagen|directorio] [config]"
    echo
    echo "Sin argumentos: corre sobre 2-CNN/test/ con el mejor modelo guardado"
    echo "modelo:         ruta al .keras (default: 2-CNN/config/models/best_model.keras)"
    echo "imagen|dir:     imagen o directorio a predecir (default: 2-CNN/test/)"
    echo "config:         ruta al YAML (default: 2-CNN/config/default.yaml)"
    echo
    echo "Ejemplos:"
    echo "  ./scripts/cnn.fish"
    echo "  ./scripts/cnn.fish 2-CNN/models/mi_modelo.keras"
    echo "  ./scripts/cnn.fish 2-CNN/models/best_model.keras rana.jpg"
    echo
    echo "Para ENTRENAR: ./scripts/trains/train_cnn.fish"
    exit 0
end

set model_path $default_model
set image_path $default_images

if test (count $argv) -ge 1
    set model_path $argv[1]
end
if test (count $argv) -ge 2
    set image_path $argv[2]
end
if test (count $argv) -ge 3
    set config_path $argv[3]
end
if test (count $argv) -ge 4
    echo "[cnn] Uso: (basename (status filename)) [modelo] [imagen|directorio] [config]" >&2
    exit 1
end

exec $python_bin "$project_root/predict.py" --config $config_path --model $model_path --image $image_path
