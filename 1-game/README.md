# Proyecto 1 - Juego con IA (Pygame + MLP)

Proyecto académico que implementa un juego simple en Pygame con recolección de datos y un modelo MLP que aprende a saltar en modo automático. El objetivo es mostrar un flujo básico de aprendizaje supervisado aplicado a un entorno de juego.

## Índice

1. [Resumen](#resumen)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Ejecución](#ejecución)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Notas](#notas)

## Resumen

- Juego en 2D con Pygame.
- Recolección de datos durante el modo manual.
- Entrenamiento de un MLP con scikit-learn.
- Modo automático que usa el modelo entrenado.

## Requisitos

- Python 3.10+ (recomendado)
- Dependencias en `requirements.txt`

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta `1-game/`:

```bash
python -m src.main
```

Alternativa directa:

```bash
python src/main.py
```

## Estructura del proyecto

```text
1-game/
├── src/
│   ├── main.py
│   ├── core/
│   ├── juego/
│   ├── ml/
│   ├── render/
│   └── ui/
├── assets/
├── juego_pygame_mlp1.py
└── requirements.txt
```

## Notas

- `juego_pygame_mlp1.py` se conserva intacto como referencia del estado original.
- La carpeta `src/` contiene la refactorización completa para mejorar mantenibilidad.
