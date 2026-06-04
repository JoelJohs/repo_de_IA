#!/usr/bin/env fish
# Autocompletado de codigo C con RNN vanilla entrenado.
# Para entrenar: ./scripts/trains/train_rnn.fish
#
# Uso:
#   ./scripts/rnn.fish --predict "int sum"               # prompt como flag
#   ./scripts/rnn.fish --predict "int sum" --max-new 80  # mas caracteres
#   echo "void swap" | ./scripts/rnn.fish --predict      # desde stdin
#   ./scripts/rnn.fish --vscode                          # abre VSCode con la extension en dev (F5)
#   ./scripts/rnn.fish --vscode prod                     # empaqueta + instala la .vsix
#   ./scripts/rnn.fish --vscode test                     # corre solo el test_e2e headless
#   ./scripts/rnn.fish -h / --help                       # esta ayuda

set script_dir   (dirname (realpath (status filename)))
set repo_root    (dirname $script_dir)
set python_bin   "$repo_root/.venv/bin/python"
set project_root "$repo_root/3-RNN"
set vscode_script "$repo_root/scripts/vscode.fish"

set model_path   "$project_root/models/rnn_v1.keras"
set meta_path    "$project_root/models/rnn_v1.meta.json"
set max_new      60
set temperature  0.4
set seed         42
set show_only_new false

if not test -x $python_bin
    echo "[rnn] No se encontro el venv en $python_bin" >&2
    echo "[rnn] Ejecuta primero: ./scripts/setup.fish" >&2
    exit 1
end

# --- parseo de flags ---------------------------------------------------------
set positional
set predict_mode false
set vscode_mode false
set vscode_arg  ""

while test (count $argv) -gt 0
    switch $argv[1]
        case --predict
            set predict_mode true
            set argv $argv[2..-1]
        case --vscode
            set vscode_mode true
            if test (count $argv) -ge 2; and string match -qrv '^-' -- $argv[2]
                set vscode_arg $argv[2]
                set argv $argv[2..-1]
            else
                set argv $argv[2..-1]
            end
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
            echo "Uso: (basename (status filename)) --predict [OPCIONES]"
            echo "     (basename (status filename)) --vscode [prod|test]"
            echo
            echo "Modo 1: prediccion por CLI"
            echo "  --predict STR      prompt a completar (o se lee de stdin)"
            echo
            echo "  Opcionales (predict):"
            echo "    --max-new N        caracteres a generar (default 60)"
            echo "    --temperature T    temperatura sampling (default 0.4; 0 = greedy)"
            echo "    --seed N           semilla aleatoria (default 42)"
            echo "    --show-only-new    imprime solo la continuacion (sin el prompt)"
            echo "    --model PATH       ruta del modelo (default models/rnn_v1.keras)"
            echo "    --meta PATH        ruta del meta.json (default models/rnn_v1.meta.json)"
            echo
            echo "Modo 2: abrir la extension VSCode en dev (F5 + Extension Dev Host)"
            echo "  --vscode           abre VSCode con la extension (dev mode, listo para F5)"
            echo "  --vscode prod      empaqueta la .vsix y la instala globalmente"
            echo "  --vscode test      corre solo el test_e2e.js (smoke test headless)"
            echo
            echo "Ejemplos:"
            echo "  ./scripts/rnn.fish --predict 'int sum'"
            echo "  ./scripts/rnn.fish --predict 'int sum' --max-new 80 --temperature 0.3"
            echo "  echo 'void swap' | ./scripts/rnn.fish --predict"
            echo "  ./scripts/rnn.fish --vscode"
            echo
            echo "Para ENTRENAR el modelo:"
            echo "  ./scripts/trains/train_rnn.fish"
            exit 0
        case '*'
            set positional $positional $argv[1]
            set argv $argv[2..-1]
    end
end

# --- modo --vscode (delega a scripts/vscode.fish) ---------------------------
if $vscode_mode
    if not test -f $vscode_script
        echo "[rnn] No se encontro $vscode_script" >&2
        exit 1
    end
    exec $vscode_script $vscode_arg
end

# --- flags excluyentes -------------------------------------------------------
if $predict_mode; and $vscode_mode
    echo "[rnn] --predict y --vscode son excluyentes. Usa uno solo." >&2
    exit 1
end

# --- sanity checks -----------------------------------------------------------
if not test -f $model_path
    echo "[rnn] No se encontro el modelo: $model_path" >&2
    echo "[rnn] Entrena primero: ./scripts/trains/train_rnn.fish" >&2
    exit 1
end

if not test -f $meta_path
    echo "[rnn] No se encontro el meta: $meta_path" >&2
    exit 1
end

# --- prompt: stdin si no vino por flag --------------------------------------
if not $predict_mode
    if isatty stdin
        echo "[rnn] Usa --predict para dar el prompt." >&2
        echo "[rnn]   ./scripts/rnn.fish --predict 'int sum'" >&2
        echo "[rnn] O desde stdin:" >&2
        echo "[rnn]   echo 'int sum' | ./scripts/rnn.fish --predict" >&2
        echo "[rnn]" >&2
        echo "[rnn] Para ENTRENAR: ./scripts/trains/train_rnn.fish" >&2
        exit 1
    end
    read prompt
else
    set prompt $positional[1]
end

if test -z "$prompt"
    echo "[rnn] Prompt vacio" >&2
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
