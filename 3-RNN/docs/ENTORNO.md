# Entorno y compatibilidad

## Por que Python 3.11.9 y no 3.14

TensorFlow 2.16.x (la version estable probada y que usa este proyecto)
**no soporta Python 3.14**. El soporte oficial llega con TF 2.20+ y
muchas piezas (Keras, TF-TRT) emiten warnings o fallan en 3.14.

| TF     | Python maximo estable |
| ------ | --------------------- |
| 2.16.x | 3.12                  |
| 2.17.x | 3.12                  |
| 2.18.x | 3.12                  |
| 2.19.x | 3.12                  |
| 2.20.x | 3.13                  |
| 2.21.x | 3.13                  |

Arch Linux trae `python 3.14.x` por defecto (muy nuevo). Para evitar
recompilar TensorFlow o instalar nightly, fijamos **3.11.9** con `pyenv`.

## Setup recomendado (Arch Linux, lo que usa este proyecto)

```bash
# pyenv ya esta instalado, y 3.11.9 ya descargado
cd 3-RNN
pyenv local 3.11.9           # crea .python-version
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import tensorflow as tf; print('TF', tf.__version__)"
# -> TF 2.16.1
```

## Diagnostico basico

```bash
python --version            # debe decir 3.11.9 (no 3.14)
which python                # debe apuntar a .venv/bin/python
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Troubleshooting

| Mensaje                                   | Causa                       | Solucion                                 |
| ----------------------------------------- | --------------------------- | ---------------------------------------- |
| `ModuleNotFoundError: tensorflow`         | El venv no esta activado    | `source .venv/bin/activate`              |
| `ImportError: numpy >= 2.0 ...`           | numpy 2.x rompe TF 2.16     | `pip install "numpy<2.0"`                |
| `Could not find cuda drivers`             | Aviso informativo, no error | Normal: este proyecto corre en CPU       |
| `TF-TRT Warning: Could not find TensorRT` | Aviso informativo           | Normal, no afecta el entrenamiento       |
| `python3.11: command not found`           | pyenv no se cargo           | `export PATH="$HOME/.pyenv/shims:$PATH"` |

## Por que CPU y no GPU

Este equipo no tiene GPU NVIDIA. TensorFlow 2.16.1 detecta la ausencia
y degrada a CPU sin fallar. El entrenamiento de 80 epocas en CPU
toma ~5 min con 20 K ventanas y `BLOCK_SIZE=32`.

Si en algun momento se agrega GPU, no se necesita tocar el codigo:
TF detecta CUDA automaticamente y mueve los tensores.

## Requirements exactos

`requirements.txt` fija:

```
tensorflow==2.16.1
numpy<2.0
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.5
```

Se instalan ademas (transitivamente): `h5py`, `keras`, `tensorboard`,
`scikit-learn-intelex` no es necesario, `matplotlib` solo se usa en el
notebook, `nbformat` y `ipykernel` solo si se quiere re-ejecutar el
notebook.
