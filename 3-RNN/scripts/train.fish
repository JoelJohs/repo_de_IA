#!/usr/bin/env fish
# Entrena el RNN vanilla de 3-RNN sobre el dataset curado.
#
# Por defecto:
#   1. Si dataset/curated/ no existe, corre curate_per_category.py
#      (50-80 funciones por categoria, heuristica estricta).
#   2. Genera las ventanas char-level con preprocess.py
#      (input = dataset/curated/funciones.c, block_size=32).
#   3. Entrena con train.py (80 epochs, Adam(1e-3), embed=48, hidden=64)
#      y guarda en models/rnn_v1.keras.
#
# Uso:
#   ./scripts/train.fish                          # flujo completo
#   ./scripts/train.fish --fresh                  # regenera curated/ y reentrena
#   ./scripts/train.fish --epochs 120             # mas epochs
#   ./scripts/train.fish --max-windows 40000      # menos sub-muestreo
#   ./scripts/train.fish --hidden 96 --embed-dim 64
#   ./scripts/train.fish --tag rnn_v2             # guardar como rnn_v2.keras
#
# Flags forwards (todos opcionales):
#   --epochs N         default 80
#   --max-windows N    default 20000   (cap de ventanas para velocidad)
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

set script_dir    (dirname (realpath (status filename)))
set project_root  (dirname $script_dir)
set repo_root     (dirname $project_root)
set python_bin    "$repo_root/.venv/bin/python"

set clean_dir     "$project_root/dataset/clean"
set curated_dir   "$project_root/dataset/curated"
set combined      "$curated_dir/funciones.c"
set processed_dir "$project_root/dataset/processed"
set models_dir    "$project_root/models"

set epochs        80
set max_windows   20000
set hidden        64
set embed_dim     48
set batch_size    128
set block_size    32
set tag           "rnn_v1"
set per_cat_min   50
set per_cat_max   80
set length_min    200
set length_max    600
set fresh         false

# --- parseo de flags ---------------------------------------------------------
set positional
while test (count $argv) -gt 0
    switch $argv[1]
        case --epochs
            set epochs $argv[2]
            set argv $argv[2..-1]
        case --max-windows
            set max_windows $argv[2]
            set argv $argv[2..-1]
        case --hidden
            set hidden $argv[2]
            set argv $argv[2..-1]
        case --embed-dim
            set embed_dim $argv[2]
            set argv $argv[2..-1]
        case --batch-size
            set batch_size $argv[2]
            set argv $argv[2..-1]
        case --block-size
            set block_size $argv[2]
            set argv $argv[2..-1]
        case --tag
            set tag $argv[2]
            set argv $argv[2..-1]
        case --per-cat-min
            set per_cat_min $argv[2]
            set argv $argv[2..-1]
        case --per-cat-max
            set per_cat_max $argv[2]
            set argv $argv[2..-1]
        case --length-min
            set length_min $argv[2]
            set argv $argv[2..-1]
        case --length-max
            set length_max $argv[2]
            set argv $argv[2..-1]
        case --fresh
            set fresh true
            set argv $argv[2..-1]
        case -h --help
            echo "Uso: (basename (status filename)) [opciones]"
            echo
            echo "Opciones principales:"
            echo "  --epochs N          epocas de entrenamiento (default 80)"
            echo "  --max-windows N     cap de ventanas (default 20000)"
            echo "  --hidden N          unidades RNN (default 64)"
            echo "  --embed-dim N       dim del embedding (default 48)"
            echo "  --batch-size N      batch size (default 128)"
            echo "  --tag NAME          nombre del modelo (default rnn_v1)"
            echo "  --fresh             regenera dataset/curated/ aunque exista"
            echo
            echo "Opciones de curacion (pasan a curate_per_category.py):"
            echo "  --per-cat-min N     (default 50)"
            echo "  --per-cat-max N     (default 80)"
            echo "  --length-min N      (default 200)"
            echo "  --length-max N      (default 600)"
            exit 0
        case '*'
            set positional $positional $argv[1]
            set argv $argv[2..-1]
    end
end

if test (count $positional) -gt 0
    echo "[train] Argumentos no reconocidos: $positional" >&2
    exit 1
end

# --- sanity checks -----------------------------------------------------------
if not test -x $python_bin
    echo "[train] No se encontro el venv unificado en $python_bin" >&2
    echo "[train] Setup: cd $repo_root && pyenv local 3.11.9 && python -m venv .venv" >&2
    exit 1
end

if not test -d $clean_dir
    echo "[train] No se encontro el dataset categorizado en $clean_dir" >&2
    exit 1
end

# --- 1) curacion -------------------------------------------------------------
if not test -d $curated_dir; or $fresh
    echo "[train] [1/3] Curando dataset (per-cat=$per_cat_min..$per_cat_max, length=$length_min..$length_max)..."
    $python_bin "$project_root/src/curate_per_category.py" \
        --clean-dir    $clean_dir \
        --out-dir      $curated_dir \
        --per-cat-min  $per_cat_min \
        --per-cat-max  $per_cat_max \
        --length-min   $length_min \
        --length-max   $length_max
    if test $status -ne 0
        echo "[train] Curacion fallo" >&2
        exit 1
    end
else
    echo "[train] [1/3] Curado: skip (dataset/curated/ ya existe; usa --fresh para regenerar)"
end

if not test -f $combined
    echo "[train] No se genero $combined" >&2
    exit 1
end
echo "[train]       corpus: $combined ("(wc -c < $combined)" bytes)"

# --- 2) preproceso -----------------------------------------------------------
echo "[train] [2/3] Preprocesando (block_size=$block_size)..."
$python_bin "$project_root/src/preprocess.py" \
    --input       $combined \
    --output-dir  $processed_dir \
    --block-size  $block_size
if test $status -ne 0
    echo "[train] Preproceso fallo" >&2
    exit 1
end

# --- 3) entrenamiento --------------------------------------------------------
echo "[train] [3/3] Entrenando $tag (epochs=$epochs, max_windows=$max_windows, hidden=$hidden, embed=$embed_dim)"
$python_bin "$project_root/src/train.py" \
    --processed-dir  $processed_dir \
    --output-dir     $models_dir \
    --epochs         $epochs \
    --max-windows    $max_windows \
    --hidden         $hidden \
    --embed-dim      $embed_dim \
    --batch-size     $batch_size \
    --tag            $tag
if test $status -ne 0
    echo "[train] Entrenamiento fallo" >&2
    exit 1
end

echo
echo "[train] OK. Modelo en: $models_dir/$tag.keras"
echo "[train] Meta:        $models_dir/$tag.meta.json"
echo "[train] History:     $models_dir/$tag.history.json"
