#!/usr/bin/env fish
# Lanza la UI PyQt6 del clasificador CNN (2-CNN).
# Misma config por default que cnn.fish, solo cambia el entry point.
#
# Uso:
#   ./scripts/cnn_ui.fish                        # config + modelo default
#   ./scripts/cnn_ui.fish ruta/a/config.yaml     # config alternativa
#   ./scripts/cnn_ui.fish -h / --help            # esta ayuda

set script_dir     (dirname (realpath (status filename)))
set repo_root      (dirname $script_dir)
set python_bin     "$repo_root/.venv/bin/python"
set project_root   "$repo_root/2-CNN"
set config_path    "$project_root/config/default.yaml"
set default_model  "$project_root/config/models/best_model.keras"

if not test -x $python_bin
    echo "[cnn_ui] No se encontro el venv en $python_bin" >&2
    echo "[cnn_ui] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

if test "$argv[1]" = "-h"; or test "$argv[1]" = "--help"
    echo "Uso: (basename (status filename)) [config]"
    echo
    echo "Abre la ventana PyQt6 para explorar el dataset, elegir imagenes"
    echo "de test/ o dataset/ y ver la prediccion vs la clase real."
    echo
    echo "config: ruta al YAML (default: 2-CNN/config/default.yaml)"
    echo "        el modelo se toma de config.models_dir/best_model.keras"
    echo
    echo "Si el modelo no existe, ver: ./scripts/trains/train_cnn.fish"
    exit 0
end

if test (count $argv) -ge 2
    echo "[cnn_ui] Uso: (basename (status filename)) [config]" >&2
    exit 1
end

if test (count $argv) -ge 1
    set config_path $argv[1]
end

if not test -f "$default_model"
    echo "[cnn_ui] No se encontro el modelo: $default_model" >&2
    echo "[cnn_ui] Entrena primero: ./scripts/trains/train_cnn.fish" >&2
    exit 1
end

cd $project_root
exec $python_bin -m src.ui.launcher --config $config_path --model $default_model
