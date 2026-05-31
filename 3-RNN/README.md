# Proyecto 3 - RNN Vanilla para Autocompletado de Codigo

Entrenar un modelo basico de lenguaje con RNN Vanilla en Keras/TensorFlow para autocompletar codigo en C y exponerlo via una API local para integracion con un editor.

## Estructura

```text
3-RNN/
├── dataset/
├── docs/
├── models/
├── notebooks/
├── src/
└── README.md
```

## Requisitos

- Python 3.10+
- Entorno virtual

## Inicializacion rapida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si TensorFlow falla en tu sistema, revisa `docs/ENTORNO.md`.

## Flujo sugerido

1. Crear el dataset con 60+ funciones en C en `dataset/`.
2. Preprocesar el dataset con scripts en `src/` o notebook.
3. Entrenar el modelo y guardar pesos en `models/`.
4. Levantar la API local para el editor.

## Notas

- La arquitectura debe usar RNN Vanilla (`SimpleRNN`).
- Evitar LSTM/GRU si el profesor exige RNN Vanilla.
