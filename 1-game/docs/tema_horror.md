# Documentacion del juego (horror religioso / psicologico)

Este documento explica el juego, el codigo y los assets necesarios para una tematica de horror religioso con tono psicologico, inspirada visualmente en personajes tipo Resident Evil.

## 1. Resumen del juego

El jugador controla a un personaje que debe sobrevivir a proyectiles mientras enfrenta entidades de horror psicologico. La estetica es oscura, con simbolos religiosos, ambientes claustrofobicos y enemigos que representan culpa, fe corrompida y delirios.

Objetivo:

- Sobrevivir esquivando proyectiles.
- Atacar y derrotar enemigos/jefes en un ciclo infinito.
- Entrenar un modelo que luego pueda jugar solo.

## 2. Como se juega (controles)

Controles propuestos:

- `ESPACIO`: saltar.
- `ABAJO` o `S`: agacharse.
- `X` o `K`: ataque frontal.
- `ESC` o `P`: volver al menu.
- `M`: modo manual (recolecta dataset).
- `A`: modo automatico (IA).
- `T`: entrenar MLP.
- `C`: exportar CSV.
- `F`: fullscreen.
- `Q`: salir.

Reglas basicas:

- Las balas pueden venir desde arriba o abajo.
- El jugador decide saltar o agacharse.
- En modo manual, cada decision se guarda en el dataset.
- En modo automatico, el modelo decide la accion.

## 3. Mecanicas principales

### 3.1 Saltar

- Mecanica actual de salto con gravedad.
- Si el jugador esta en el aire, no puede iniciar otro salto.

### 3.2 Agacharse (requerido)

- Reduce la altura del personaje para esquivar balas medias.
- Debe ajustar la hitbox del jugador.

### 3.3 Balas arriba/abajo (requerido)

- Las balas tienen tres alturas posibles: arriba, medio, abajo.
- Esto obliga a alternar entre salto y agacharse.

### 3.4 Ataque frontal (mejora personal)

- Permite golpear a un enemigo a la derecha.
- Requiere un estado `atacando` con duracion corta.

### 3.5 Enemigos y jefes infinitos (mejora personal)

- Un enemigo fijo a la derecha con barra de vida.
- Cada jefe aumenta su vida en el orden: 1, 1.5, 2, 4, 8, y luego duplicar.
- El juego no termina: el jugador avanza a jefes infinitos.

## 4. Arquitectura del codigo

Rutas clave (relativas a `1-game/`):

- `src/main.py`: punto de entrada.
- `src/juego/juego.py`: loop del juego, estados y reglas.
- `src/render/activos.py`: carga y escalado de assets.
- `src/render/dibujar.py`: funciones de dibujo.
- `src/ui/menu.py`: menu y mensajes.
- `src/ml/`: dataset, entrenamiento, decision automatica.
- `src/core/`: constantes y tipos.

## 5. Funcionamiento tecnico del codigo

### 5.1 Loop principal

`Juego.loop()`:

1. Muestra menu.
2. Procesa input.
3. Registra dataset (manual) o decide accion (IA).
4. Actualiza salto/agacharse.
5. Mueve la bala.
6. Dibuja escena.

### 5.2 Registro del dataset

El dataset se guarda en memoria como lista de `Sample`.

Variables necesarias para la nueva mecanica:

- `velocidad_bala`
- `distancia`
- `salto`
- `agachado`
- `bala_y` o `altura_bala`

Esto permite que la IA aprenda a diferenciar entre saltar o agacharse.

### 5.3 Entrenamiento del modelo

El MLP usa:

- `train_test_split` estratificado.
- `StandardScaler`.
- `MLPClassifier`.

Si el dataset es de una sola clase, se usa un modelo trivial.

### 5.4 Decision automatica

La decision debe evolucionar a una accion:

- `saltar`
- `agacharse`
- `nada`

Esto se puede implementar con una salida multiclasica o con reglas simples sobre la probabilidad.

## 6. Assets necesarios (tematica horror religioso)

### 6.1 Personaje (estilo Resident Evil)

- `assets/sprites/mono_frame_1.png`
- `assets/sprites/mono_frame_2.png`
- `assets/sprites/mono_frame_3.png`
- `assets/sprites/mono_frame_4.png`
- `assets/sprites/mono_crouch.png` (agachado)
- `assets/sprites/mono_attack.png` (ataque)

Estetica sugerida:

- Silueta humana con abrigo o atuendo oscuro.
- Rasgos realistas, paleta apagada.

### 6.2 Enemigos (horror psicologico)

- `assets/sprites/enemy.png`
- `assets/sprites/boss.png` (opcional)

Ideas visuales:

- Figuras angelicales corrompidas.
- Marionetas religiosas deformadas.
- Entidades con halos rotos y simbolos blasfemos.

### 6.3 Fondo y ambiente

- `assets/game/fondo2.png` (reemplazar por ambiente oscuro)

Ideas:

- Catedral en ruinas.
- Pasillos con velas y vitrales rotos.
- Oscuridad con niebla y luz puntual.

### 6.4 Proyectiles

- `assets/sprites/purple_ball.png` (puede reemplazarse por orbes o espinas)
- Variantes opcionales:
  - `assets/sprites/purple_ball_up.png`
  - `assets/sprites/purple_ball_down.png`

### 6.5 UI y barra de vida

- `assets/ui/hp_bar.png` (opcional)

Si no hay sprite, la barra puede dibujarse por codigo.

## 7. Direccion artistica

Paleta sugerida:

- Base: negros, grises frios, marrones oscuros.
- Acentos: rojo profundo, dorado envejecido.
- Luz: velas o halos rotos en amarillo tenue.

Tono psicologico:

- Simbolos religiosos distorsionados.
- Escenarios opresivos.
- Enemigos que representen culpa y delirio.

## 8. Checklist de entrega

- [ ] Agacharse implementado (input y hitbox).
- [ ] Balas arriba y abajo.
- [ ] Dataset actualizado con `agachado` y `bala_y`.
- [ ] Exportacion CSV con nuevas columnas.
- [ ] IA entrenable con dataset nuevo.

## 9. Checklist de mejoras personales

- [ ] Ataque frontal.
- [ ] Enemigo fijo con barra de vida.
- [ ] Jefes infinitos con vida escalada.
- [ ] Puntaje por dano y derrotas.
