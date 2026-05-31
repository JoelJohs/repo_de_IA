# Entorno y compatibilidad

Si TensorFlow se comporta raro, esta guia te ayuda a estabilizar el entorno.

## Diagnostico basico

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Recomendaciones generales

- Usa un entorno virtual limpio por proyecto.
- Evita mezclar pip con paquetes del sistema.
- Si usas Arch, prefiere Python 3.10 o 3.11 para TensorFlow.

## Opciones comunes

### Opcion A: Python 3.11 en venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Opcion B: Miniconda (si el sistema rompe pip)

1. Instalar Miniconda.
2. Crear entorno:

```bash
conda create -n rnn python=3.11
conda activate rnn
pip install -r requirements.txt
```

## Notas

- Si tienes GPU NVIDIA, revisa compatibilidad de CUDA con TensorFlow.
- Para CPU-only, TensorFlow funciona bien sin configuracion extra.
