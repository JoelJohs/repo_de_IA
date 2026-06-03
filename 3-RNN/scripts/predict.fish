#!/usr/bin/env fish
# Ejecuta el modelo RNN entrenado para autocompletar una linea de codigo C.
# Imprime prompt + continuacion (o solo la continuacion con --show-only-new).
#
# Uso:
#   ./scripts/predict.fish "int sum"
#   ./scripts/predict.fish "int sum" --max-new 80 --temperature 0.3
#   echo "int sum" | ./scripts/predict.fish
#   ./scripts/predict.fish --show-only-new "void swap(int"
#
# Flags:
#   --prompt STR        texto a completar (o se lee de stdin si falta)
#   --max-new N         chars a generar (default 60)
#   --temperature T     temperatura del sampling (default 0.4; 0 = greedy)
#   --seed N            semilla aleatoria (default 42)
#   --model PATH        ruta del modelo (default models/rnn_v1.keras)
#   --meta PATH         ruta del meta.json (default models/rnn_v1.meta.json)
#   --show-only-new     imprime SOLO la continuacion (sin el prompt)
#   -h / --help         muestra esta ayuda

set script_dir    (dirname (realpath (status filename)))
set project_root  (dirname $script_dir)
set repo_root     (dirname $project_root)
set python_bin    "$repo_root/.venv/bin/python"

set model_path    "$project_root/models/rnn_v1.keras"
set meta_path     "$project_root/models/rnn_v1.meta.json"
set prompt        ""
set max_new       60
set temperature   0.4
set seed          42
set show_only_new false

# --- parseo de flags ---------------------------------------------------------
set positional
while test (count $argv) -gt 0
    switch $argv[1]
        case --prompt
            set prompt $argv[2]
            set argv $argv[2..-1]
        case --max-new --max_new
            set max_new $argv[2]
            set argv $argv[2..-1]
        case --temperature
            set temperature $argv[2]
            set argv $argv[2..-1]
        case --seed
            set seed $argv[2]
            set argv $argv[2..-1]
        case --model
            set model_path $argv[2]
            set argv $argv[2..-1]
        case --meta
            set meta_path $argv[2]
            set argv $argv[2..-1]
        case --show-only-new
            set show_only_new true
            set argv $argv[2..-1]
        case -h --help
            echo "Uso: (basename (status filename)) [opciones] [prompt]"
            echo
            echo "Si prompt no se pasa con --prompt, se toma del primer argumento"
            echo "posicional o de stdin."
            echo
            echo "Opciones:"
            echo "  --prompt STR        texto a completar"
            echo "  --max-new N         chars a generar (default 60)"
            echo "  --temperature T     temperatura (default 0.4; 0 = greedy)"
            echo "  --seed N            semilla aleatoria (default 42)"
            echo "  --model PATH        ruta del modelo .keras"
            echo "  --meta PATH         ruta del meta.json"
            echo "  --show-only-new     imprime solo la continuacion"
            exit 0
        case '*'
            set positional $positional $argv[1]
            set argv $argv[2..-1]
    end
end

# --- sanity checks -----------------------------------------------------------
if not test -x $python_bin
    echo "[predict] No se encontro el venv unificado en $python_bin" >&2
    echo "[predict] Setup: cd $repo_root && pyenv local 3.11.9 && python -m venv .venv" >&2
    exit 1
end

if not test -f $model_path
    echo "[predict] No se encontro el modelo: $model_path" >&2
    echo "[predict] Entrenalo primero: ./scripts/train.fish" >&2
    exit 1
end

if not test -f $meta_path
    echo "[predict] No se encontro el meta: $meta_path" >&2
    exit 1
end

# --- prompt: --prompt > primer posicional > stdin ---------------------------
if test -z "$prompt"
    if test (count $positional) -ge 1
        set prompt $positional[1]
    else
        read prompt
    end
end

if test -z "$prompt"
    echo "[predict] Prompt vacio" >&2
    exit 1
end

# --- ejecucion --------------------------------------------------------------
set extra_args
if $show_only_new
    set extra_args --show-only-new
end

exec $python_bin "$project_root/src/predict.py" \
    --model       $model_path \
    --meta        $meta_path \
    --prompt      "$prompt" \
    --max-new     $max_new \
    --temperature $temperature \
    --seed        $seed \
    $extra_args
