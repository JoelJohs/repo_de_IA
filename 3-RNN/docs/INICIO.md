# Guia de inicio

## Objetivo

Entrenar un modelo RNN Vanilla para autocompletar codigo en C e integrarlo con un editor via una API local.

## Dependencias sugeridas

- tensorflow
- numpy
- pandas
- scikit-learn
- matplotlib
- jupyter
- fastapi
- uvicorn

## Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Dataset

- Colocar un archivo con 60+ funciones en C dentro de `dataset/`.
- Mantener estilo consistente para que el modelo aprenda patrones.
- Ver detalles en `docs/DATASET.md`.

## Entrenamiento

- Preprocesar por caracteres o tokens.
- Usar `SimpleRNN` para cumplir la practica.

## API local (editor)

- Levantar un servidor sencillo con FastAPI.
- Consumirlo desde el editor elegido.
