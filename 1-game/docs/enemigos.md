# Enemigo y jefes infinitos

Este documento describe la mejora personal de combate con enemigo fijo y jefes con vida escalada.

## Enemigo fijo a la derecha

### Idea base

- Un enemigo permanece en el lado derecho de la pantalla.
- Tiene una barra de vida visible.
- Recibe dano cuando el jugador ataca.

### Datos minimos

- `enemigo_x`, `enemigo_y`.
- `enemigo_vida`.
- `enemigo_vida_max`.
- `enemigo_activo`.

## Barra de vida

### Opcion 1: dibujar por codigo

```python
def dibujar_barra_vida(pantalla, x, y, vida, vida_max):
    ancho_total = 120
    alto = 12
    porcentaje = max(0.0, min(1.0, vida / vida_max))
    ancho_actual = int(ancho_total * porcentaje)

    pygame.draw.rect(pantalla, (80, 80, 80), (x, y, ancho_total, alto))
    pygame.draw.rect(pantalla, (200, 60, 60), (x, y, ancho_actual, alto))
```

### Opcion 2: sprite

Usar `assets/ui/hp_bar.png` y recortar segun el porcentaje.

## Ataque del jugador

### Mecanica propuesta

- Al presionar `X` (o `K`), se activa `atacando` por un tiempo corto.
- Se crea un rect de ataque frente al jugador.
- Si colisiona con el enemigo, se resta vida.

```python
if self.atacando:
    ataque_rect = pygame.Rect(
        self.jugador.right,
        self.jugador.y + 10,
        int(40 * self.scale),
        int(20 * self.scale),
    )
    if ataque_rect.colliderect(self.enemigo_rect):
        self.enemigo_vida -= 1
```

## Sistema de jefes infinitos

### Escalado solicitado

Vida por jefe (simplificado):

1. 1
2. 1.5
3. 2
4. 4
5. 8
6. Luego duplicar siempre

### Implementacion sugerida

```python
def vida_jefe_por_nivel(nivel: int) -> float:
    if nivel == 1:
        return 1
    if nivel == 2:
        return 1.5
    if nivel == 3:
        return 2
    if nivel == 4:
        return 4
    if nivel == 5:
        return 8
    return 8 * (2 ** (nivel - 5))
```

### Flujo

1. Al derrotar un jefe, `nivel_jefe += 1`.
2. Se reinicia `enemigo_vida` con `vida_jefe_por_nivel`.
3. Se incrementa puntaje.

## Puntaje

### Sugerencia

- +10 por golpe.
- +100 por derrotar un jefe.
- Bonus por racha de golpes.

El puntaje puede mostrarse en el HUD junto con datos del modelo.
