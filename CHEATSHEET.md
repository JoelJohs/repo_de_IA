# Cheat Sheet — Comandos rápidos

Python 3.11.9 · venv: `.venv/` · sin GPU · shell: fish

```bash
source .venv/bin/activate      # activar entorno (bash/zsh)
source .venv/bin/activate.fish # activar entorno (fish)
./scripts/setup.fish           # setup completo (pyenv → venv → deps)
```

---

## 1-game — Juego + MLP

```
./scripts/game.fish
```

Controles: `M` manual · `A` automático · `T` entrenar · `C` exportar CSV · `Q` salir

---

## 2-CNN — CNN animales

```
./scripts/cnn.fish                                    # predecir todo test/
./scripts/cnn.fish 2-CNN/models/best_model.keras rana.jpg
./scripts/cnn_ui.fish                                 # GUI PyQt6 (explorar + predecir)
./scripts/trains/train_cnn.fish                       # entrenar
```

Entrenar requiere dataset en `2-CNN/dataset/`. Construirlo:
`python -m 2-CNN.src.data.build_processed_dataset --config 2-CNN/config/default.yaml`

---

## 3-RNN — Autocompletado código C

```
./scripts/rnn.fish --predict "int sum"
./scripts/rnn.fish --predict "int sum" --max-new 80
echo "void swap" | ./scripts/rnn.fish --predict
./scripts/rnn.fish --vscode                          # abre VSCode (F5) con la extension en dev
./scripts/rnn.fish --vscode prod                     # empaqueta + instala la .vsix
./scripts/rnn.fish --vscode test                     # smoke test headless
./scripts/trains/train_rnn.fish                       # entrenar (~5 min CPU)
```

API: `uvicorn 3-RNN.src.api:app --port 8000`
VS Code (alias equivalente): `./scripts/vscode.fish`

---

## 4-RAG — RAG seguridad pública México

```
ollama pull qwen2.5:7b-instruct           # requerido una vez

./scripts/rag.fish --rag                  # RAG interactivo
./scripts/rag.fish --evaluate             # métricas de recuperación
./scripts/rag.fish --api                  # FastAPI :8000
./scripts/rag.fish --generate-behavioral  # 300 ejemplos comportamentales
./scripts/rag.fish --finetune             # LoRA Qwen0.5B (CPU lento)
./scripts/rag.fish --integrate            # RAG + LoRA
./scripts/rag.fish --compare both --save  # antes/después
./scripts/rag.fish --official             # 10 preguntas RAG 2026
./scripts/rag.fish --help                 # todos los flags
```

---

## Jupyter

```
./scripts/jupyter.fish      # abre en navegador → notebooks/index.ipynb
./scripts/jupyter.fish lab  # JupyterLab
```

Notebooks: `notebooks/index.ipynb` (landing page con los 7 notebooks)

---

## Resumen de scripts

| Script | Acción |
|---|---|
| `setup.fish` | Setup inicial (pyenv, venv, pip install) |
| `game.fish` | Ejecutar juego 1-game |
| `cnn.fish` | Inferencia CNN |
| `cnn_ui.fish` | GUI PyQt6 (explorar + predecir) |
| `rnn.fish` | Predicción RNN |
| `rag.fish` | Pipeline RAG completo |
| `jupyter.fish` | Lanzar Jupyter |
| `vscode.fish` | Extensión VS Code (3-RNN) |
| `trains/train_cnn.fish` | Entrenar CNN |
| `trains/train_rnn.fish` | Entrenar RNN (curado → preproceso → fit) |
