#!/usr/bin/env fish
# Entrena el RNN vanilla de 3-RNN para autocompletado de codigo C.
# Pipeline completo: curacion → preproceso → entrenamiento.
# Delegates a 3-RNN/scripts/train.fish.
#
# Uso:
#   ./scripts/trains/train_rnn.fish                          # flujo completo
#   ./scripts/trains/train_rnn.fish --fresh                  # regenera dataset/curated/
#   ./scripts/trains/train_rnn.fish --epochs 120             # mas epochs
#   ./scripts/trains/train_rnn.fish --max-windows 40000      # mas ventanas
#   ./scripts/trains/train_rnn.fish --tag rnn_v2             # guardar como rnn_v2
#
# Flags (forward a 3-RNN/scripts/train.fish):
#   --epochs N         default 80
#   --max-windows N    default 20000
#   --hidden N         default 64
#   --embed-dim N      default 48
#   --batch-size N     default 128
#   --block-size N     default 32
#   --tag NAME         default rnn_v1
#   --per-cat-min N    default 50
#   --per-cat-max N    default 80
#   --length-min N     default 200
#   --length-max N     default 600
#   --fresh            fuerza regeneracion de dataset/curated/

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname (dirname $script_dir))
set python_bin   "$repo_root/.venv/bin/python"
set project_root "$repo_root/3-RNN"

if not test -x $python_bin
    echo "[train_rnn] No se encontro el venv en $python_bin" >&2
    echo "[train_rnn] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

if not test -d "$project_root/dataset/clean"
    echo "[train_rnn] No se encontro el dataset categorizado en $project_root/dataset/clean" >&2
    exit 1
end

# Forward all args to the existing train script
exec "$project_root/scripts/train.fish" $argv
