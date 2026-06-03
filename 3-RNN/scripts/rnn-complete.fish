#!/usr/bin/env fish
# Completa una linea de codigo C usando el modelo RNN entrenado.
# Imprime SOLO la continuacion por stdout (sin el prefijo).
#
# Uso:
#   ./scripts/rnn-complete.fish "int sum"
#   echo "int sum" | ./scripts/rnn-complete.fish
#   cat codigo.c | ./scripts/rnn-complete.fish   # lee la primera linea
#
# Opciones:
#   -n N    cantidad de caracteres a generar (default: 60)
#   -t T    temperatura del sampling (default: 0.4; 0 = greedy)
#   -s S    semilla aleatoria (default: 42)
#
# Requisitos:
#   - IA/.venv (venv unificado en la raiz del repo) con tensorflow instalado
#   - models/rnn_v1.keras entrenado (python src/train.py --epochs 80)

set script_dir (dirname (realpath (status filename)))
set project_root (dirname $script_dir)
set repo_root (dirname $project_root)
set python_bin "$repo_root/.venv/bin/python"
set model_path "$project_root/models/rnn_v1.keras"
set meta_path  "$project_root/models/rnn_v1.meta.json"

set max_new 60
set temperature 0.4
set seed 42

# Parsea flags
set positional
while test (count $argv) -gt 0
    switch $argv[1]
        case -n --max-new
            set max_new $argv[2]
            set argv $argv[2..-1]
        case -t --temperature
            set temperature $argv[2]
            set argv $argv[2..-1]
        case -s --seed
            set seed $argv[2]
            set argv $argv[2..-1]
        case -h --help
            echo "Uso: (basename (status filename)) [opciones] \"prompt\""
            echo
            echo "Opciones:"
            echo "  -n N    caracteres a generar (default: 60)"
            echo "  -t T    temperatura del sampling (default: 0.4; 0 = greedy)"
            echo "  -s S    semilla aleatoria (default: 42)"
            exit 0
        case '*'
            set positional $positional $argv[1]
            set argv $argv[2..-1]
    end
end

# Verifica venv
if not test -x $python_bin
    echo "[rnn-complete] No se encontro el venv unificado en $python_bin" >&2
    echo "[rnn-complete] Setup: cd $repo_root && pyenv local 3.11.9 && python -m venv .venv" >&2
    exit 1
end

# Verifica modelo
if not test -f $model_path
    echo "[rnn-complete] No se encontro el modelo: $model_path" >&2
    echo "[rnn-complete] Entrena primero: python src/train.py --epochs 80" >&2
    exit 1
end

# Obtiene el prompt
set prompt
if test (count $positional) -ge 1
    set prompt $positional[1]
else
    read prompt
end

if test -z "$prompt"
    echo "[rnn-complete] Prompt vacio" >&2
    exit 1
end

# Llama a predict.py con --show-only-new para imprimir SOLO la continuacion
exec $python_bin "$project_root/src/predict.py" \
    --model $model_path \
    --meta $meta_path \
    --prompt "$prompt" \
    --max-new $max_new \
    --temperature $temperature \
    --seed $seed \
    --show-only-new
