#!/usr/bin/env fish
# Curar (o regenerar) el dataset de funciones C a partir de dataset/clean/.
# Sin entrenar, solo escribe dataset/curated/<categoria>.c + funciones.c + REPORTE.md.
#
# Uso:
#   ./scripts/curate.fish                    # curar con defaults (50-80/cat)
#   ./scripts/curate.fish --per-cat-max 100  # mas funciones por categoria
#   ./scripts/curate.fish --length-max 800   # permitir funciones mas largas
#   ./scripts/curate.fish --fresh            # alias para forzar regeneracion
#
# Flags (todos opcionales):
#   --per-cat-min N     default 50
#   --per-cat-max N     default 80
#   --length-min N      default 200
#   --length-max N      default 600
#   --min-returns N     default 1
#   --min-control N     default 1

set script_dir    (dirname (realpath (status filename)))
set project_root  (dirname $script_dir)
set repo_root     (dirname $project_root)
set python_bin    "$repo_root/.venv/bin/python"

set clean_dir     "$project_root/dataset/clean"
set curated_dir   "$project_root/dataset/curated"

set per_cat_min   50
set per_cat_max   80
set length_min    200
set length_max    600
set min_returns   1
set min_control   1
set fresh         false

# --- parseo ------------------------------------------------------------------
while test (count $argv) -gt 0
    switch $argv[1]
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
        case --min-returns
            set min_returns $argv[2]
            set argv $argv[2..-1]
        case --min-control
            set min_control $argv[2]
            set argv $argv[2..-1]
        case --fresh
            set fresh true
            set argv $argv[2..-1]
        case -h --help
            echo "Uso: (basename (status filename)) [opciones]"
            echo "  --per-cat-min N     (default 50)"
            echo "  --per-cat-max N     (default 80)"
            echo "  --length-min N      (default 200)"
            echo "  --length-max N      (default 600)"
            echo "  --min-returns N     (default 1)"
            echo "  --min-control N     (default 1)"
            echo "  --fresh             regenera aunque curated/ exista"
            exit 0
        case '*'
            echo "[curate] Flag no reconocida: $argv[1]" >&2
            exit 1
    end
end

# --- sanity ------------------------------------------------------------------
if not test -x $python_bin
    echo "[curate] No se encontro el venv unificado en $python_bin" >&2
    exit 1
end

if not test -d $clean_dir
    echo "[curate] No se encontro $clean_dir" >&2
    exit 1
end

if test -d $curated_dir; and not $fresh
    echo "[curate] $curated_dir ya existe. Usa --fresh para regenerar." >&2
    exit 1
end

# --- ejecucion ---------------------------------------------------------------
echo "[curate] Curando (per-cat=$per_cat_min..$per_cat_max, length=$length_min..$length_max)..."
$python_bin "$project_root/src/curate_per_category.py" \
    --clean-dir    $clean_dir \
    --out-dir      $curated_dir \
    --per-cat-min  $per_cat_min \
    --per-cat-max  $per_cat_max \
    --length-min   $length_min \
    --length-max   $length_max \
    --min-returns  $min_returns \
    --min-control  $min_control
