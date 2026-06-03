#!/usr/bin/env fish
# Menu interactivo para los 4 proyectos de IA.
# Uso: ./scripts/menu.fish
# (C) 2026 — Materia Inteligencia Artificial

set script_dir (dirname (realpath (status filename)))
set repo_root  (dirname $script_dir)

# ---- colores ----
set c_title  (set_color -o cyan)
set c_opt    (set_color yellow)
set c_desc   (set_color white)
set c_dim    (set_color brblack)
set c_err    (set_color red)
set c_ok     (set_color green)
set c_warn   (set_color magenta)
set c_reset  (set_color normal)

# ---- helpers ----
function check_model
    switch $argv[1]
        case game
            test -f "$repo_root/1-game/data/game_data.csv"
        case cnn
            test -f "$repo_root/2-CNN/config/models/best_model.keras"
        case rnn
            test -f "$repo_root/3-RNN/models/rnn_v1.keras"
        case rag
            test -f "$repo_root/4-RAG/vectordb/chroma.sqlite3"
    end
end

function status_icon
    if check_model $argv[1]
        echo "$c_ok  ✅$c_reset"
    else
        echo "$c_dim  ⏳$c_reset"
    end
end

function note_if_missing
    if not check_model $argv[1]
        echo "$c_dim (entrenar primero)$c_reset"
    end
end

function run_script
    set script "$repo_root/scripts/$argv[1].fish"
    if not test -f $script
        echo "$c_err[menu] Script no encontrado: $script$c_reset"
        sleep 2
        return 1
    end
    eval $script $argv[2..-1]
    if test $status -ne 0
        echo
        echo "$c_err[menu] Terminó con código $status.$c_reset"
    end
    echo
    echo "$c_desc Presiona ENTER para volver al menú...$c_reset"
    read -l dummy
end

function run_notebook
    set nb "$repo_root/notebooks/$argv[1].ipynb"
    if not test -f $nb
        echo "$c_err[menu] Notebook no encontrado: $nb$c_reset"
        sleep 2
        return 1
    end
    "$repo_root/.venv/bin/jupyter" notebook "$nb"
end

# ──────────────────────────────────────────────
# PANTALLA PRINCIPAL
# ──────────────────────────────────────────────
function main_screen
    clear
    echo "$c_title╔══════════════════════════════════════════════════════════╗$c_reset"
    echo "$c_title║           INTELIGENCIA ARTIFICIAL — Proyectos          ║$c_reset"
    echo "$c_title║  1-game · 2-CNN · 3-RNN · 4-RAG Seguridad México      ║$c_reset"
    echo "$c_title╚══════════════════════════════════════════════════════════╝$c_reset"
    echo
    echo "  $c_opt 0$c_reset  🔧 Inicializar todo (venv + dependencias)"
    echo
    echo "  ─── Modelos entrenados ───$c_opt"
    echo "  $c_opt 1$c_reset  🎮 Juego + MLP$(status_icon game)"
    echo "       $c_dim Demostración de perceptrón multicapa. Recolectas"
    echo "       puntos haciendo clic, la red clasifica la trayectoria.$c_reset"
    echo
    echo "  $c_opt 2$c_reset  🐾 CNN Clasificación Animales$(status_icon cnn)"
    echo "       $c_dim Red convolucional que distingue 5 especies:"
    echo "       rana·araña·mono·ballena·pájaro — precisión 76%$c_reset"
    echo
    echo "  $c_opt 3$c_reset  ⌨️  RNN Autocompletado C$(status_icon rnn)"
    echo "       $c_dim Red recurrente char-level que completa código"
    echo "       fuente en C. Entrenada con 60+ funciones reales.$c_reset"
    echo
    echo "  $c_opt 4$c_reset  📚 RAG Seguridad Pública MX$(status_icon rag)"
    echo "       $c_dim Pipeline de Retrieval-Augmented Generation sobre"
    echo "       23 documentos oficiales. ChromaDB + all-MiniLM + Ollama.$c_reset"
    echo
    echo "  ─── Utilidades ───"
    echo "  $c_opt 5$c_reset  📖 CHEATSHEET — comandos rápidos"
    echo "  $c_opt 6$c_reset  📓 Notebooks Jupyter (landing page)"
    echo "  $c_opt q$c_reset  $c_err Salir$c_reset"
    echo
    echo -n "  $c_opt >>>$c_reset "
end

# ──────────────────────────────────────────────
# SUB-MENÚ 1-GAME
# ──────────────────────────────────────────────
function game_menu
    while true
        clear
        echo "$c_title       🎮 JUEGO + PERCEPTRÓN MULTICAPA (1-game)$c_reset"
        echo "$c_title  ─────────────────────────────────────────────$c_reset"
        echo
        echo "  Proyecto: Demostración de MLP con sklearn."
        echo "  El jugador mueve el mouse y hace clic para recolectar"
        echo "  puntos. Al terminar, el MLP aprende a clasificar si la"
        echo "  trayectoria fue 'buena' o 'mala' según los clics."
        echo
        set game_status "$c_ok+ Entrenado ✅$c_reset"
        if not check_model game
            set game_status "$c_warn+ Sin datos aún$c_reset"
        end
        echo "  Estado: $game_status"
        echo
        echo "  $c_opt 1$c_reset  JUGAR — abre ventana Pygame, recolecta datos,"
        echo "       entrena el MLP online y exporta CSV"
        echo "  $c_opt 2$c_reset  JUGAR con trayectoria pre-hecha (demo rápida)"
        echo "  $c_opt 3$c_reset  Notebook: análisis del MLP entrenado"
        echo "  $c_opt B$c_reset  ← Volver"
        echo
        echo -n "  $c_opt >>>$c_reset "
        read -l opt
        switch $opt
            case 1; run_script game
            case 2; run_script game --pre-recorded
            case 3; run_notebook 1-game_analisis
            case b B; return 0
            case '*'
                echo "$c_err  Opción inválida$c_reset"; sleep 1
        end
    end
end

# ──────────────────────────────────────────────
# SUB-MENÚ 2-CNN
# ──────────────────────────────────────────────
function cnn_menu
    while true
        clear
        echo "$c_title       🐾 CNN CLASIFICACIÓN DE ANIMALES (2-CNN)$c_reset"
        echo "$c_title  ─────────────────────────────────────────────$c_reset"
        echo
        echo "  Red convolucional (Keras/TF) que clasifica imágenes"
        echo "  en 5 clases: rana · araña · mono · ballena · pájaro"
        echo "  Arquitectura: Conv2D → MaxPool → Dropout → Dense"
        echo "  Precisión en test: 76% (ballenas 5/5, ranas 2/5)"
        echo
        set cnn_model "$c_ok  ✅ best_model.keras$c_reset"
        if not check_model cnn
            set cnn_model "$c_err  ❌ no encontrado$c_reset"
        end
        echo "  Modelo entrenado:$cnn_model"
        echo
        echo "  $c_opt 1$c_reset  PREDECIR sobre test/ completo $(note_if_missing cnn)"
        echo "       Evalúa el modelo contra todas las imágenes en test/"
        echo
        echo "  $c_opt 2$c_reset  PREDECIR una imagen específica"
        echo "       Te pide la ruta y muestra top-3 predicciones"
        echo
        echo "  $c_opt 3$c_reset  ENTRENAR modelo desde cero"
        echo "       Corre 30 épocas con data augmentation en train/"
        echo
        echo "  $c_opt 4$c_reset  Notebook: entrenamiento y arquitectura"
        echo "  $c_opt 5$c_reset  Notebook: inferencia y resultados"
        echo "  $c_opt B$c_reset  ← Volver"
        echo
        echo -n "  $c_opt >>>$c_reset "
        read -l opt
        switch $opt
            case 1; run_script cnn
            case 2
                echo
                echo -n "  Ruta de la imagen: "
                read -l img_path
                run_script cnn "" "$img_path"
            case 3; run_script trains/train_cnn
            case 4; run_notebook 2-CNN_entrenamiento
            case 5; run_notebook 2-CNN_inferencia
            case b B; return 0
            case '*'
                echo "$c_err  Opción inválida$c_reset"; sleep 1
        end
    end
end

# ──────────────────────────────────────────────
# SUB-MENÚ 3-RNN
# ──────────────────────────────────────────────
function rnn_menu
    while true
        clear
        echo "$c_title       ⌨️  RNN AUTOCOMPLETADO DE CÓDIGO C (3-RNN)$c_reset"
        echo "$c_title  ─────────────────────────────────────────────$c_reset"
        echo
        echo "  Red recurrente vanilla (char-level) con Keras."
        echo "  Entrenada con 60+ funciones extraídas de código real."
        echo "  Dado un prompt como 'int sum(', genera la continuación"
        echo "  carácter por carácter con sampling controlado."
        echo
        set rnn_status "$c_ok  ✅ rnn_v1.keras$c_reset"
        if not check_model rnn
            set rnn_status "$c_err  ❌ no encontrado$c_reset"
        end
        echo "  Modelo entrenado:$rnn_status"
        echo
        echo "  $c_opt 1$c_reset  PREDECIR — escribe un prompt en C $(note_if_missing rnn)"
        echo "       Ej: 'int suma', 'void swap(int', 'for (int i = 0'"
        echo "       Parámetros: max-new (60), temperatura (0.4)"
        echo
        echo "  $c_opt 2$c_reset  ENTRENAR modelo desde cero"
        echo "       Pipeline: curar corpus → preprocesar → entrenar 80 epochs"
        echo
        echo "  $c_opt 3$c_reset  Notebook: entrenamiento y análisis"
        echo "  $c_opt 4$c_reset  Notebook: inferencia interactiva"
        echo "  $c_opt B$c_reset  ← Volver"
        echo
        echo -n "  $c_opt >>>$c_reset "
        read -l opt
        switch $opt
            case 1
                echo
                echo -n "  Prompt de código C: "
                read -l prompt
                run_script rnn --predict "$prompt"
            case 2; run_script trains/train_rnn
            case 3; run_notebook 3-RNN_entrenamiento
            case 4; run_notebook 3-RNN_inferencia
            case b B; return 0
            case '*'
                echo "$c_err  Opción inválida$c_reset"; sleep 1
        end
    end
end

# ──────────────────────────────────────────────
# SUB-MENÚ 4-RAG
# ──────────────────────────────────────────────
function rag_menu
    while true
        clear
        echo "$c_title       📚 RAG — SEGURIDAD PÚBLICA EN MÉXICO (4-RAG)$c_reset"
        echo "$c_title  ─────────────────────────────────────────────$c_reset"
        echo
        echo "  Pipeline completo de Retrieval-Augmented Generation:"
        echo "   1. Chunking de 23 PDFs oficiales → 7,595 fragmentos"
        echo "   2. Embeddings con all-MiniLM-L6-v2 → ChromaDB"
        echo "   3. Generación aumentada con Ollama (qwen2.5:7b)"
        echo "  Evaluación: P@5=0.26, R@5=0.53, category match=0.7"
        echo
        set rag_status "$c_ok  ✅ vectordb con 7,595 chunks$c_reset"
        if not check_model rag
            set rag_status "$c_warn  ⏳ reconstruir primero (opción 2)$c_reset"
        end
        echo "  Índice:$rag_status"
        echo
        echo "  $c_opt 1$c_reset  CHATEAR — modo interactivo con el RAG"
        echo "       Carga vectordb (~8s), luego conversación libre."
        echo "       Comandos dentro del chat: /fuentes, /modelo, /salir"
        echo
        echo "  $c_opt 2$c_reset  Reconstruir índice (chunking + embeddings)"
        echo "       Lee los 23 PDFs desde cero y regenera ChromaDB."
        echo "       Necesario la primera vez o si cambian los PDFs."
        echo
        echo "  $c_opt 3$c_reset  Responder las 10 preguntas oficiales"
        echo "       Evalúa el RAG contra preguntas de 3 niveles"
        echo "       (básico, intermedio, avanzado) con sus fuentes."
        echo
        echo "  $c_opt 4$c_reset  Iniciar API REST (FastAPI en :8000)"
        echo "       Sirve el RAG como endpoint HTTP."
        echo
        echo "  $c_opt 5$c_reset  Notebook: pipeline RAG completo"
        echo "  $c_opt B$c_reset  ← Volver"
        echo
        echo -n "  $c_opt >>>$c_reset "
        read -l opt
        switch $opt
            case 1; run_script rag
            case 2; run_script rag --rebuild
            case 3; run_script rag --official
            case 4; run_script rag --api
            case 5; run_notebook 4-RAG_completo
            case b B; return 0
            case '*'
                echo "$c_err  Opción inválida$c_reset"; sleep 1
        end
    end
end

# ──────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────
while true
    main_screen
    read -l opt

    switch $opt
        case 0
            run_script setup
        case 1; game_menu
        case 2; cnn_menu
        case 3; rnn_menu
        case 4; rag_menu
        case 5
            echo
            if test -f "$repo_root/CHEATSHEET.md"
                bat -p --color=always "$repo_root/CHEATSHEET.md" 2>/dev/null; or cat "$repo_root/CHEATSHEET.md"
            else
                echo "$c_err CHEATSHEET.md no encontrado$c_reset"
            end
            echo
            echo "$c_desc Presiona ENTER para volver...$c_reset"
            read -l dummy
        case 6
            run_notebook index
        case q Q
            echo
            echo "$c_ok  ¡Hasta luego!$c_reset"
            exit 0
        case '*'
            echo "$c_err  Opción inválida: $opt$c_reset"
            sleep 1
    end
end
