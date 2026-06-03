# Guia de inicio rapido

```bash
# El venv esta unificado en la raiz del repo (IA/.venv)
cd ../..
source .venv/bin/activate.fish

# (re)generar todo desde cero
python src/sample_clean.py
python src/preprocess.py
python src/train.py --epochs 80
```

Listo. Tenes:

- `models/rnn_v1.keras` con el modelo (~244 KB).
- `dataset/sample/funciones.c` con las 150 funciones seleccionadas.
- `notebooks/entrenamiento.ipynb` ejecutado, con perdida y ejemplos.

## Probar el modelo

### CLI

```bash
python src/predict.py --prompt "int sum" --max-new 60 --temperature 0.4
```

### API HTTP (FastAPI)

```bash
uvicorn src.api:app --port 8000
```

En otra terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt":"void print","max_new":40,"temperature":0.4}'
```

### Servidor stdio (VS Code)

```bash
echo '{"id":1,"method":"complete","prefix":"int sum","max_new":40}' | \
     python -m src.server_stdio
```

## VS Code

```bash
code vscode-extension/
```

Presiona F5 dentro de la carpeta de la extension → se abre una ventana
de Extension Development Host con la extension cargada. En cualquier
archivo `.c`, Ctrl+Shift+Space dispara la autocompletacion.

Si queres empaquetar la extension para instalarla en tu VS Code real:

```bash
cd vscode-extension
npm install -g @vscode/vsce
vsce package
code --install-extension rnn-c-autocomplete-0.1.0.vsix
```

## Que hace cada script

| Script                          | Funcion                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| `src/sample_clean.py`           | Lee `dataset/funciones.c`, extrae 150 funciones ASCII sin `int main()` y con estilo consistente. |
| `src/preprocess.py`             | Vocabulario char-level + ventanas de 32 chars. Salida en `dataset/processed/`.                   |
| `src/train.py`                  | Define la arquitectura, entrena 80 epocas, guarda `rnn_v1.keras` + meta.                         |
| `src/predict.py`                | Carga el modelo y genera continuaciones autorregresivas.                                         |
| `src/api.py`                    | FastAPI con `POST /predict` y `POST /suggest`.                                                   |
| `src/server_stdio.py`           | JSON-line por stdin/stdout para la extension VS Code.                                            |
| `notebooks/entrenamiento.ipynb` | Mismo flujo que los scripts, en formato notebook, ejecutado.                                     |

## Problemas frecuentes

| Problema                              | Solucion                                                                                                    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: tensorflow`     | Activa el venv raiz: `cd ../.. && source .venv/bin/activate.fish`                                          |
| `python: command not found`           | El venv no se cargo. Reactivar.                                                                             |
| Perdida no baja                       | El corpus se reescribio. Volve a correr `python src/preprocess.py && python src/train.py`.                  |
| La extension VS Code no conecta       | El path de `serverScript` o `pythonPath` en settings.json no es correcto. Ver `vscode-extension/README.md`. |
| Version de Python incompatible con TF | Ver `docs/ENTORNO.md` - usar pyenv 3.11.9.                                                                  |
