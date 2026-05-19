# AGENT - Proyecto 2-CNN

## Objetivo

Proyecto de clasificacion de imagenes de animales (ranas, aranas, pajaros, ballenas, changos) usando CNN con TensorFlow. Los notebooks entregados por el profesor son **referenciales** y no deben modificarse.

Referencias:
- `2-CNN/CNN.ipynb`
- `2-CNN/CNNriesgo.ipynb`

## Reglas del proyecto

- No modificar los notebooks.
- Usar dataset por carpetas (clasificacion). No hay bounding boxes.
- Mantener compatibilidad con CPU (sin GPU).
- Mantener el CNN base del profesor como modelo principal.
- Documentar decisiones en `2-CNN/README.md` y `2-CNN/.opencode/CONTEXT.md`.

## Flujo de trabajo recomendado

1. Ajustar configuracion en `config/default.yaml`.
2. Entrenar con `python train.py --config config/default.yaml`.
3. Guardar modelo y clases en `models/` y `artifacts/`.
4. Predecir con `python predict.py --config config/default.yaml --model models/best_model.keras --image ruta/a/imagen.jpg`.
5. (Pendiente) Implementar GUI en PyQt6.
