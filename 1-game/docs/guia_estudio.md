# Guia de estudio — 1-game

Documento corto para defender el proyecto en clase. Cubre arquitectura, flujo,
datos, modelo y las preguntas tipicas que hace el profesor.

## 1. Arbol de archivos

```
src/
├── main.py              # punto de entrada
├── core/
│   ├── constantes.py    # BASE_W/H, escala, colores, DURACION_AGACHAR
│   └── tipos.py         # Sample (fila del dataset)
├── juego/
│   ├── juego.py         # Juego: orquesta todo (loop, estados, render, IA)
│   ├── loop.py          # procesa eventos del teclado
│   ├── fisica.py        # reglas de movimiento (salto, agacharse)
│   ├── bala.py          # disparo, movimiento y reset
│   └── estado.py        # dataclasses: PlayerState, BulletState, etc.
├── ml/
│   ├── dataset.py       # registra muestras y exporta CSV
│   ├── modelo.py        # MLPClassifier
│   ├── arbol.py         # DecisionTreeClassifier
│   └── visualizacion.py # graficos 2D/3D
├── render/
│   ├── activos.py       # carga y escala sprites
│   └── dibujar.py       # dibuja fondo, jugador, bala, nave
└── ui/
    └── menu.py          # menu principal
```

## 2. Flujo del loop (en `juego.py` -> `loop`)

Cada frame, en este orden:

1. **Input** — `procesar_eventos()` lee teclas y llama callbacks.
2. **IA** — en modo auto, `decision_auto_*` devuelve accion 0, 1 o 2.
3. **Update** — `_update_background`, `_update_bullet`, `_update_player`, colision.
4. **Render** — dibuja fondo, jugador, bala, nave, HUD, flip.

`Juego` solo orquesta; no dibuja pixeles ni decide reglas.

## 3. Responsabilidad por archivo (1 linea)

| Archivo | Que hace |
|---|---|
| `main.py` | arranca el juego |
| `juego/juego.py` | loop + estado + conexiones |
| `juego/loop.py` | traduce teclas en callbacks |
| `juego/fisica.py` | reglas de movimiento (puras) |
| `juego/bala.py` | trayectoria y respawn de balas |
| `juego/estado.py` | dataclasses de estado |
| `ml/dataset.py` | guardar muestras + CSV |
| `ml/modelo.py` | entrenar y predecir con MLP |
| `ml/arbol.py` | entrenar y predecir con Arbol |
| `ml/visualizacion.py` | graficos del dataset |
| `render/activos.py` | cargar sprites y escalarlos |
| `render/dibujar.py` | blits a pantalla |
| `ui/menu.py` | menu principal |

## 4. Como se almacenan los datos

- `ModelState.datos: List[Sample]` vive en memoria dentro de `Juego`.
- `Sample` (en `core/tipos.py`) tiene 8 campos por frame:
  - `velocidad_bala`, `distancia`, `bala_y`, `bala_arriba` -> estado del mundo
  - `salto` (0/1), `agachado` (0/1) -> estado del jugador
  - `accion` (0/1/2) -> lo que decidio el jugador
  - `ataque_color` (0/1) -> tipo de bala (alias de `bala_arriba`)
- Se registra una muestra por frame en modo manual con
  `registrar_decision_manual()`. El CSV sale con `C` desde el menu.

**Punto clave para el profesor:** la etiqueta `accion` solo se guarda durante
los frames en los que el jugador REALMENTE esta saltando o agachado:

- `accion = 1` mientras `player.salto == True` (el salto dura lo que dura la
  parabola).
- `accion = 2` mientras `player.agachado == True` (el agachado dura
  `DURACION_AGACHAR` frames).

Por eso un press de DOWN produce ~10 filas con `accion=2`, no un bloque
infinito. El modelo aprende el patron, no el "mantener la tecla".

## 5. Salto y agacharse: el mismo patron

Ambos son **impulsos con duracion**:

- **Salto** — `iniciar_salto()` activa `player.salto=True`. `aplicar_salto()`
  aplica gravedad cada frame hasta volver al suelo y desactiva la bandera.
  La duracion depende de la fisica (gravedad + velocidad inicial).
- **Agacharse** — `iniciar_agacharse()` solo enciende la bandera
  `player.agachado` y carga `agachado_timer = DURACION_AGACHAR`. No mueve
  ni el `x` ni el `y`, ni cambia la altura; el cambio es **solo
  visual** (en render se blittea `down.png` en la misma posicion que el
  sprite de caminar). `actualizar_agacharse()` decrementa el timer y, al
  llegar a 0, apaga la bandera y se vuelve a blittear el sprite normal.

Mismo patron de input (un KEYDOWN dispara todo), misma forma de etiqueta
(0/1/2 segun frames activos). Por eso saltan y se agachan igual de bien.

## 6. Modo auto (IA jugando)

- `decision_auto_saltar()` y `decision_auto_arbol()` reciben el estado actual
  y devuelven `accion` (0/1/2) y `proba_salto` (probabilidad de la clase 1).
- Pipeline: `decidir_accion_auto()` -> si `accion==1` se llama `iniciar_salto`,
  si `accion==2` se llama `_iniciar_impulso_agacharse_auto`. Esta funcion
  NO refresca el timer: el agachado dura exactamente `DURACION_AGACHAR`
  frames y se levanta solo. Para encadenar otro agachado la IA debe volver
  a votar `2` cuando el timer esta en 0. Asi la IA replica la cadencia
  del jugador: si el jugador oprime DOWN 5 veces rapido, el dataset tiene
  5 impulsos encadenados con gaps minimos, y la IA vota `2` siguiendo el
  mismo patron. No se "estanca" agachada.
- `_estilo_hist` (deque de 200) detecta el estilo dominante del jugador
  manual y lo prioriza si supera el 55% con >= 40 muestras. Si el estilo
  dominante es agacharse, la IA vota `2` casi siempre, lo que produce
  impulsos encadenados -> el personaje aparenta estar siempre agachado
  pero respetando la duracion del impulso (el ojo no lo nota a 45 FPS).

## 7. Entrenamiento

- `entrenar_modelo()` (MLP) y `entrenar_arbol()` (Arbol):
  - Si hay < 80 muestras -> modelo trivial que siempre devuelve la clase
    unica (0 = nada).
  - Si hay una sola clase -> modelo trivial "SIEMPRE X".
  - Si hay datos de >= 2 clases -> `train_test_split` (80/20) +
    `StandardScaler` + `MLPClassifier` o `DecisionTreeClassifier`.
  - Devuelve accuracy en test.
- Las features son: `velocidad_bala, distancia, bala_y, bala_arriba,
  ataque_color`.
- La etiqueta es `accion`.

## 7b. MLP en detalle

Configuracion exacta en `src/ml/modelo.py:57-63`:

- **Activacion en capas ocultas: `relu`.** ReLU es `f(x) = max(0, x)`. No
  satura para valores positivos, asi el gradiente fluye sin achicarse.
  Sigmoide y tanh saturan (salida casi plana) y producen *vanishing
  gradient* — problema serio en redes profundas, molesto incluso en 2
  capas chicas.
- **Activacion de salida: softmax (implicito).** `MLPClassifier` aplica
  softmax sobre las clases; por eso `predict_proba` devuelve probabilidades
  que suman 1. Para 2 clases softmax es matematicamente equivalente a
  sigmoide; para 3 clases (0/1/2) hace falta softmax. Por eso no hay
  "sigmoide" aca: la salida es multiclase.
- **Optimizador: `adam`.** Ajusta el learning rate por parametro usando
  momentos (media y varianza moviles de los gradientes). Converge rapido
  sin tunear lr a mano. Alternativas que NO usamos: `sgd` (lento, sensible
  al lr) y `lbfgs` (quasi-Newton, mas memoria, no aporta con dataset
  chico).
- **Loss: cross-entropia logistica** (por defecto en `MLPClassifier`),
  coherente con softmax en la salida.
- **Arquitectura `(3, 3)`.** 2 capas ocultas, 3 neuronas cada una.
  Entradas = 5 features, salida = 3 clases. Conteo de parametros
  entrenables: entrada → h1 = 5·3 + 3 = 18; h1 → h2 = 3·3 + 3 = 12;
  h2 → salida = 3·3 + 3 = 12. **Total ≈ 42 parametros.** Pocos a proposito:
  con dataset chico, mas capacidad sobreajusta.
- **`StandardScaler` previo.** Media 0, desvio 1. Sin el scaler,
  `velocidad_bala` (decenas) y `distancia` (centenas) dominan sobre
  `bala_arriba` y `ataque_color` (0/1). Los pesos del MLP se desbalancean
  y la red aprende casi solo de las features grandes.
- **`train_test_split(stratify=y, test_size=0.2, random_state=42)`.**
  Estratificar mantiene la proporcion de clases en train y test;
  `random_state=42` fija la semilla para reproducibilidad (mismo split,
  misma inicializacion de pesos, mismo accuracy).
- **Backprop.** Lo hace `sklearn` por dentro, no hay loop manual. Nosotros
  solo llamamos `clf.fit(X_train, y_train)`.
- **`max_iter=300000`.** Tope de iteraciones del optimizador. No entrena
  "mejor" cuanto mas alto: sklearn corta por convergencia (loss bajo la
  tolerancia). El numero grande es solo un techo por si tarda en convergir.

## 7c. Arbol en detalle

Configuracion exacta en `src/ml/arbol.py:51`:

- **Sin funcion de activacion.** No es red neuronal. Cada nodo divide
  con un umbral del estilo `feature <= threshold`. Por eso el arbol no
  usa ReLU, sigmoide ni nada: es un clasificador basado en reglas.
- **Hiperparametros `max_depth=6, min_samples_leaf=4`.** Limitan la
  profundidad y exigen minimo de muestras por hoja. Podan el arbol para
  evitar sobreajuste con dataset chico.
- **No usa scaler.** Los arboles son invariantes a transformaciones
  monotonas de una feature (umbral se reescala solo). Aplica tanto al
  Arbol como a Random Forest y Gradient Boosting.
- **`predict_proba` devuelve frecuencias** relativas de la clase en la
  hoja a la que cayo la muestra. No hay softmax ni calibracion.
- **Diferencia clave con el MLP.** El arbol produce reglas legibles
  (`distancia < X AND bala_y > Y -> salta`); el MLP produce pesos que no
  se interpretan directamente. Por eso el arbol sirve para "explicar" la
  politica y el MLP para jugarla con precision.

## 8. Controles

| Tecla | Accion |
|---|---|
| ESPACIO | saltar (manual) |
| ABAJO / S | agacharse impulso (manual) |
| M | modo manual (reinicia dataset y modelo) |
| A | auto MLP |
| T | entrenar MLP |
| R | auto Arbol |
| D | entrenar Arbol |
| C | exportar CSV |
| F | fullscreen toggle |
| P / ESC | volver al menu |
| Q | salir |

La ventana es **resizable**: arrastra la esquina para redimensionar y todo
se reescala. F alterna pantalla completa.

## 9. Preguntas tipo examen (respuesta corta)

**Arquitectura**
1. *Cual es el flujo principal?* Input -> IA -> Update -> Render, dentro de
   `Juego.loop()`.
2. *Que hace `Juego` y que NO debe hacer?* Orquesta, no decide reglas ni
   dibuja pixeles.
3. *Donde esta el punto de entrada?* `src/main.py`.
4. *Por que separar estado y reglas?* Para explicar cada parte por separado
   y mezclar lo menos posible logica con render.

**Jugador y movimiento**
5. *Cuando se puede saltar?* Solo si `en_suelo` y NO agachado.
6. *Que pasa al terminar el salto?* Vuelve a `ground_y`, `salto=False`,
   `en_suelo=True`.
7. *Como se agacha ahora?* Un KEYDOWN dispara un impulso de
   `DURACION_AGACHAR` frames; al expirar se levanta solo. Es solo un
   cambio de sprite (`down.png`) en la misma posicion; ni `x`, ni `y`,
   ni altura cambian.
8. *Por que no es toggle?* Porque el toggle hacia que el modelo aprendiera
   a "agacharse siempre" al mantener la tecla.

**Bala**
9. *Cuando se dispara?* Cuando no hay bala activa (`disparada=False`).
10. *Como se decide la altura?* Aleatoria (arriba o abajo) en `bala.py`.
11. *Que pasa si la bala sale de pantalla?* `reset_bala()` la devuelve al
    borde derecho.

**IA y dataset**
12. *Que guarda el dataset?* velocidad, distancia, salto, bala_y,
    bala_arriba, agachado, accion, ataque_color.
13. *Cuantas muestras para entrenar?* >= 80 y al menos 2 clases.
14. *Que pasa con una sola clase?* Modelo trivial "SIEMPRE X".
15. *Que hace el scaler?* Normaliza features para el MLP.
16. *Que es `proba_salto`?* Probabilidad estimada de la clase 1.
17. *Por que la IA puede imitar estilos malos?* `_estilo_hist` prioriza la
    accion dominante si supera 55% con >= 40 muestras.
17b. *La IA se queda agachada para siempre si vota 2?* No. El modo auto NO
    refresca el timer: cada voto `2` produce un impulso de
    `DURACION_AGACHAR` frames, despues el personaje se levanta. La IA debe
    votar `2` de forma continua para encadenar impulsos, replicando
    exactamente la cadencia del jugador.

**Render y UI**
18. *Donde se cargan los assets?* `src/render/activos.py`.
19. *Que muestra el HUD en auto?* `proba_salto` y `accion` actual.
20. *Que pasa al redimensionar la ventana?* `_on_resize` recalcula escala y
    reposiciona jugador, nave y bala.

## 9b. Preguntas trampa del compañero (T1–T12)

Estas son las que mezcla con las reales para ver si distinguen detalle de
trampa. Formato: pregunta / trampa / respuesta corta.

| # | Pregunta del compañero | Trampa detectada | Respuesta corta |
|---|---|---|---|
| T1 | "Donde esta el sigmoide?" | Asume que hay uno. | No usamos sigmoide en las hidden. Usamos **ReLU** (`activation="relu"` en `modelo.py:59`). En la salida va softmax implicito, no sigmoide. |
| T2 | "Si no usas sigmoide, que usas y para que sirve?" | Doble pregunta + asume que deberia haber sigmoide. | **ReLU** en las dos capas ocultas `(3, 3)`. Sirve para meter no-linealidad sin saturar el gradiente. La salida es multiclase (0/1/2) asi que el softmax implicito de sklearn es el reemplazo natural del sigmoide. |
| T3 | "Y si saco ReLU y pongo activacion lineal?" | Prueba si sabes que rompe. | La red colapsa a una composicion de lineales = una sola lineal. Solo podria separar con un hiperplano. Pierde la capacidad de modelar la region del estilo "saltar si bala baja y cerca". Accuracy cae fuerte. |
| T4 | "Que optimizador y por que no otro?" | Quiere que nombres alternativas. | **Adam** — adapta el learning rate por parametro con momentos. NO usamos `sgd` (lento y sensible al lr) ni `lbfgs` (mas memoria, no aporta con dataset chico). |
| T5 | "Hacen backprop a mano?" | Quiere ver un loop de gradientes. | No. `MLPClassifier` de sklearn lo hace internamente. Nosotros solo llamamos `clf.fit(X_train, y_train)`. |
| T6 | "Por que `random_state=42`?" | Suena a numero magico. | Reproducibilidad. Fija la semilla del `train_test_split` y de la inicializacion de pesos del MLP. Si lo cambias, el split y los pesos iniciales cambian, y con ellos el accuracy reportado. |
| T7 | "Por que accuracy y no F1?" | Trampa: clases desbalanceadas. | Con `stratify=y` la proporcion se mantiene en train y test, asi que accuracy es razonable. F1 seria mejor si hubiera desbalance fuerte (p.ej. 95% clase 0) — ahi accuracy engaña. |
| T8 | "Si saco el scaler que pasa?" | Trampa. | Las features grandes (`velocidad_bala`, `distancia`) dominan los pesos; la red aprende casi solo de ellas e ignora `bala_y`, `bala_arriba` y `ataque_color`. Accuracy cae. |
| T9 | "El arbol tiene funcion de activacion?" | Trampa conceptual (mezcla modelos). | No. No es red neuronal. Cada nodo divide con un umbral `feature <= threshold`. Por eso no usa ReLU ni sigmoide ni nada. |
| T10 | "Cuantos parametros entrenables tiene el MLP?" | Trampa: hay que contar. | Entrada 5 → h1: 5·3 + 3 = **18**. h1 → h2: 3·3 + 3 = **12**. h2 → salida 3: 3·3 + 3 = **12**. **Total ≈ 42 parametros.** |
| T11 | "Si subo `max_iter` a 1.000.000 entrena mejor?" | Trampa. | No infinitamente. sklearn corta por convergencia (loss bajo la tolerancia). `max_iter` es un techo, no una mejora. |
| T12 | "Por que `stratify=y`?" | Suena a detalle menor. | Mantiene la proporcion de clases en train y test. Sin el, un test del 20% sobre un dataset chico puede quedar sin clase 1 o sin clase 2 y el accuracy reportado engaña. |

## 9c. Como responder una pregunta trampa

Cuatro pasos:

1. **Detectar la asuncion oculta.** La mayoria asume algo que no es cierto
   en el proyecto (que hay sigmoide, que accuracy es siempre suficiente,
   que el backprop lo escribimos a mano).
2. **Negar o matizar primero.** Decir "no, en este proyecto X" *antes* de
   responder lo concreto. Eso desarma la trampa.
3. **Responder lo concreto** con dato del codigo: nombre del parametro,
   archivo y linea cuando se pueda.
4. **Anclar con ejemplo del propio proyecto** para que la respuesta no
   quede en abstracto.

Ejemplo aplicado a T1 *"Donde esta el sigmoide?"*:

> "No usamos sigmoide en las capas ocultas. En el MLP usamos **ReLU** —
> esta en `modelo.py:59` como `activation="relu"` — porque no satura y
> entrena mas rapido con nuestro dataset chico. En la salida va softmax
> implicito (sklearn lo hace solo), y como nuestra salida es multiclase
> (0/1/2), softmax es lo que corresponde, no sigmoide."
