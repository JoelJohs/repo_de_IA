# Guias de defensa oral

Esta carpeta contiene las guias detalladas para defender cada proyecto
del curso frente al profesor. Cada guia sigue el mismo formato:

- Resumen ejecutivo de 1 carilla (las preguntas mas probables)
- Recorrido completo del pipeline (de la entrada a la salida)
- Detalle por archivo del codigo (con `archivo:linea`)
- Explicaciones puntuales del procesamiento
- Trampas conocidas / objeciones tipicas del profesor
- Glosario de terminos tecnicos en lenguaje sencillo

## Guias disponibles

| Proyecto | Archivos | Cubre |
|---|---|---|
| 2-CNN | [GUIA_CNN.md](./GUIA_CNN.md) | CNN clasificador de animales (5 clases) |
| 3-RNN | [GUIA_RNN.md](./GUIA_RNN.md) | RNN vanilla char-level + extension VS Code |
| 4-RAG | [GUIA_RAG.md](./GUIA_RAG.md) + [GUIA_RAG_COLAB.md](./GUIA_RAG_COLAB.md) | RAG + LoRA sobre Qwen2.5-0.5B (defensa + migracion a Colab) |
| 1-game | _(pendiente)_ | MLP del juego + pygame |

## Como usar estas guias

1. **Lectura previa** (antes de la defensa): lee el resumen ejecutivo y
   el glosario. Si te queda tiempo, lee el recorrido del pipeline.
2. **Durante la defensa**: si el profe te pregunta algo puntual, busca
   en "Trampas conocidas" — probablemente este ahi.
3. **Para profundizar**: si no entendes algo, busca el `archivo:linea`
   exacto en el codigo real del repo y abrilo.

## Notas especiales

- **GUIA_RAG_COLAB.md** NO es una guia de defensa. Es una guia de
  migracion a Google Colab (recomendada por el profe para el Proyecto
  4) con snippets copy-paste para correr RAG + LoRA en GPU T4.
