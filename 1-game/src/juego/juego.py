from __future__ import annotations

import os
from collections import deque
from typing import Deque, Optional, Tuple

import pygame

from core.constantes import BASE_H, BASE_W, EXTRA_SCALE, WINDOW_FRACTION
from ml.dataset import exportar_datos_csv, registrar_decision_manual
from ml.arbol import decision_auto_arbol, entrenar_arbol
from ml.modelo import decision_auto_saltar, entrenar_modelo
from ml.visualizacion import graficar_datos_2d, graficar_datos_3d
from render.activos import cargar_activos
from render.dibujar import (
    dibujar_bala,
    dibujar_fondo,
    dibujar_info_modelo,
    dibujar_nave,
    dibujar_personaje,
)
from ui.menu import dibujar_menu

from .bala import disparar_bala, mover_bala, reset_bala
from .estado import (
    AnimationState,
    BulletState,
    ModelState,
    PlayerState,
    RenderState,
)
from .fisica import actualizar_agacharse, aplicar_salto, iniciar_agacharse, iniciar_salto
from .loop import procesar_eventos


class Juego:
    def __init__(self) -> None:
        pygame.init()
        pygame.mixer.init()
        self._audio_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "audio"
        )

        self._flags = pygame.RESIZABLE
        self._fullscreen = False

        start_w, start_h = self._fit_window_to_screen()
        self.pantalla = pygame.display.set_mode(
            (start_w, start_h), self._flags
        )
        pygame.display.set_caption("Juego: Bala + salto + MLP (solo memoria)")

        self.corriendo = True
        self.modo_auto = False
        self._auto_modelo = "mlp"

        self.model = ModelState(
            datos=[],
            modelo_mlp=None,
            scaler=None,
            entrenado_mlp=False,
            clase_unica_mlp=None,
            modelo_arbol=None,
            entrenado_arbol=False,
            clase_unica_arbol=None,
            ultima_proba=None,
            accion_auto="none",
        )

        self._accion_manual = 0
        self._estilo_hist: Deque[int] = deque(maxlen=200)
        self._estilo_min_muestras = 40
        self._estilo_umbral = 0.55

        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin_x = 50
        self.margin_top = 100
        self.ground_y = self.h - 100
        self.player_size = (60, 96)
        self.bullet_size = (64, 64)
        self.ship_size = (192, 192)
        self.fondo_speed = 3

        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0

        self.anim = AnimationState(current_frame=0, frame_speed=10, frame_count=0)
        self.render = RenderState(fondo_x1=0, fondo_x2=start_w)

        self.mostrar_hitbox = False
        self._hitbox_bala_scale = 0.4

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    def _fit_window_to_screen(self) -> Tuple[int, int]:
        try:
            info = pygame.display.Info()
            max_w = info.current_w or BASE_W
            max_h = info.current_h or BASE_H
        except Exception:
            return BASE_W, BASE_H
        target_ratio = BASE_W / BASE_H
        screen_ratio = max_w / max_h
        if screen_ratio >= target_ratio:
            h = max(1, int(max_h * WINDOW_FRACTION))
            w = max(BASE_W // 2, int(h * target_ratio))
        else:
            w = max(1, int(max_w * WINDOW_FRACTION))
            h = max(BASE_H // 2, int(w / target_ratio))
        return w, h

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = (self.w / BASE_W) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin_x = int(self.w * 0.06)
        self.margin_top = int(self.h * 0.10)
        ground_offset = int(100 * self.scale)
        ground_adjust = int(100 * self.scale)
        self.ground_y = self.h - ground_offset - ground_adjust

        self.player_size = (int(60 * self.scale), int(96 * self.scale))
        self.bullet_size = (int(64 * self.scale), int(64 * self.scale))
        self.ship_size = (int(64 * 3 * self.scale), int(64 * 3 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.salto_vel_inicial *= 1.15
        self.gravedad = 1 * self.scale

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(21 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(16 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(
                self.margin_x, self.ground_y, self.player_size[0], self.player_size[1]
            )
            self.bala = pygame.Rect(
                self.w - self.margin_x - self.bullet_size[0],
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0],
                self.bullet_size[1],
            )
            nave_y = self.ground_y + self.player_size[1] - self.ship_size[1]
            nave_x = self.w - self.ship_size[0] - self.margin_x
            self.nave = pygame.Rect(nave_x, nave_y, self.ship_size[0], self.ship_size[1])

        self.player = PlayerState(
            rect=self.jugador,
            salto=False,
            en_suelo=True,
            salto_vel=self.salto_vel_inicial,
            agachado=False,
            agachado_timer=0,
        )
        self.bullet = BulletState(
            rect=self.bala,
            velocidad=int(-10 * self.scale),
            disparada=False,
            arriba=False,
            dist_min=None,
        )

    def _cargar_assets(self) -> None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assets = cargar_activos(
            base,
            self.player_size,
            self.bullet_size,
            self.ship_size,
            (self.w, self.h),
        )
        self.jugador_frames = assets["jugador_frames"]
        self.jugador_jump = assets["jugador_jump"]
        self.jugador_down = assets["jugador_down"]
        self.bala_img_arriba = assets["bala_img_arriba"]
        self.bala_img_abajo = assets["bala_img_abajo"]
        self.fondo_img = assets["fondo_img"]
        self.nave_img = assets["nave_img"]

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            w, h = self._fit_window_to_screen()
            self.pantalla = pygame.display.set_mode((w, h), self._flags)
            self._apply_resolution(w, h, reset_positions=True)
        self._reset_estado_juego()

    def _on_resize(self, w: int, h: int) -> None:
        w = max(320, int(w))
        h = max(240, int(h))
        self.pantalla = pygame.display.set_mode((w, h), self._flags)
        had_positions = hasattr(self, "jugador")
        self._apply_resolution(w, h, reset_positions=not had_positions)
        if had_positions:
            self.jugador.x = self.margin_x
            self.jugador.y = self.ground_y
            self.jugador.height = self.player_size[1]
            self.nave.x = self.w - self.ship_size[0] - self.margin_x
            self.nave.y = self.ground_y + self.player_size[1] - self.ship_size[1]
            self.bala.x = self.w - self.margin_x - self.bullet_size[0]
            self.bala.y = self.ground_y + int(10 * self.scale)
            self.render.fondo_x1 = 0

    def _reset_estado_juego(self) -> None:
        self.jugador.x = self.margin_x
        self.jugador.y = self.ground_y
        self.jugador.height = self.player_size[1]
        self.nave.x = self.w - self.ship_size[0] - self.margin_x
        self.nave.y = self.ground_y + self.player_size[1] - self.ship_size[1]
        self.bala.x = self.w - self.margin_x - self.bullet_size[0]
        self.bala.y = self.ground_y + int(10 * self.scale)

        self.player.salto = False
        self.player.en_suelo = True
        self.player.salto_vel = self.salto_vel_inicial
        self.player.agachado = False
        self.player.agachado_timer = 0

        self.bullet.disparada = False
        self.bullet.velocidad = int(-10 * self.scale)
        self.bullet.arriba = False
        self.bullet.dist_min = None

        self.render.fondo_x1 = 0
        self.render.fondo_x2 = self.w
        self._decision_frame_counter = 0
        self._accion_manual = 0

    def _musica_menu(self) -> None:
        ruta = os.path.join(self._audio_base, "menu.mp3")
        if os.path.isfile(ruta):
            pygame.mixer.music.load(ruta)
            pygame.mixer.music.play(-1)

    def _musica_game(self) -> None:
        ruta = os.path.join(self._audio_base, "game.mp3")
        if os.path.isfile(ruta):
            pygame.mixer.music.load(ruta)
            pygame.mixer.music.play(-1)

    def _musica_stop(self) -> None:
        pygame.mixer.music.stop()

    def _reset_modelo(self) -> None:
        self.model.modelo_mlp = None
        self.model.scaler = None
        self.model.entrenado_mlp = False
        self.model.clase_unica_mlp = None
        self.model.modelo_arbol = None
        self.model.entrenado_arbol = False
        self.model.clase_unica_arbol = None
        self.model.ultima_proba = None
        self.model.accion_auto = "none"
        self._estilo_hist.clear()
        self._accion_manual = 0

    def _calcular_hitbox_bala(self) -> pygame.Rect:
        size = max(1, int(self.bullet_size[0] * self._hitbox_bala_scale))
        offset = (self.bullet_size[0] - size) // 2
        return pygame.Rect(
            self.bala.x + offset,
            self.bala.y + offset,
            size,
            size,
        )

    def exportar_datos_csv(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return exportar_datos_csv(self.model.datos, base)

    def graficar_datos_2d(self) -> str:
        return graficar_datos_2d(self.model.datos)

    def graficar_datos_3d(self) -> str:
        return graficar_datos_3d(self.model.datos)

    def registrar_decision_manual(self) -> None:
        accion = (
            1
            if self.player.salto or self._accion_manual == 1
            else 2
            if self.player.agachado
            else 0
        )
        ataque_color = 1 if self.bullet.arriba else 0
        self._estilo_hist.append(accion)
        registrar_decision_manual(
            self.model.datos,
            self.jugador.x,
            self.bala.x,
            self.bala.y,
            self.player.en_suelo,
            self.player.agachado,
            accion,
            self.bullet.velocidad,
            self.bullet.arriba,
            ataque_color,
        )
        self._accion_manual = 0

    def _obtener_accion_estilo(self) -> Optional[int]:
        if len(self._estilo_hist) < self._estilo_min_muestras:
            return None
        conteos = {0: 0, 1: 0, 2: 0}
        for accion in self._estilo_hist:
            conteos[accion] += 1
        accion_dom = max(conteos, key=conteos.get)
        ratio = conteos[accion_dom] / len(self._estilo_hist)
        if ratio >= self._estilo_umbral:
            return accion_dom
        return None

    def entrenar_modelo(self) -> Tuple[bool, str]:
        modelo, scaler, clase_unica, mensaje = entrenar_modelo(self.model.datos)
        if modelo is None and clase_unica is None:
            return False, mensaje

        self._reset_modelo()
        self.model.modelo_mlp = modelo
        self.model.scaler = scaler
        self.model.entrenado_mlp = True
        self.model.clase_unica_mlp = clase_unica
        return True, mensaje

    def entrenar_arbol(self) -> Tuple[bool, str]:
        modelo, clase_unica, mensaje = entrenar_arbol(self.model.datos)
        if modelo is None and clase_unica is None:
            return False, mensaje

        self._reset_modelo()
        self.model.modelo_arbol = modelo
        self.model.entrenado_arbol = True
        self.model.clase_unica_arbol = clase_unica
        return True, mensaje

    def _decidir_accion_auto(self) -> Tuple[int, Optional[float]]:
        if self._auto_modelo == "arbol":
            if not self.model.entrenado_arbol:
                return 0, None
            accion, proba_salto = decision_auto_arbol(
                self.model.modelo_arbol,
                self.model.clase_unica_arbol,
                self.bullet.disparada,
                self.player.en_suelo,
                self.jugador.x,
                self.bala.x,
                self.bala.y,
                self.bullet.velocidad,
                self.bullet.arriba,
            )
        else:
            if not self.model.entrenado_mlp:
                return 0, None
            accion, proba_salto = decision_auto_saltar(
                self.model.modelo_mlp,
                self.model.scaler,
                self.model.clase_unica_mlp,
                self.bullet.disparada,
                self.player.en_suelo,
                self.jugador.x,
                self.bala.x,
                self.bala.y,
                self.bullet.velocidad,
                self.bullet.arriba,
            )
        self.model.ultima_proba = proba_salto
        accion_estilo = self._obtener_accion_estilo()
        if accion_estilo is not None:
            accion = accion_estilo
        self.model.accion_auto = (
            "salta" if accion == 1 else "agacha" if accion == 2 else "none"
        )
        return accion, proba_salto

    def mostrar_menu(self) -> None:
        self._musica_menu()
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            dibujar_menu(
                self.pantalla,
                self.fuente,
                self.fuente_chica,
                self.w,
                self.h,
                self.scale,
                len(self.model.datos),
                self.model.entrenado_mlp,
                self.model.entrenado_arbol,
                self._auto_modelo,
                self.decision_window,
                msg,
            )
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.model.datos.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.model.entrenado_mlp:
                            msg = "Primero entrena el MLP (T) en esta sesion."
                        else:
                            self.modo_auto = True
                            self._auto_modelo = "mlp"
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_r:
                        if not self.model.entrenado_arbol:
                            msg = "Primero entrena el Arbol (D) en esta sesion."
                        else:
                            self.modo_auto = True
                            self._auto_modelo = "arbol"
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_d:
                        ok, info = self.entrenar_arbol()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    def _update_animation(self) -> None:
        self.anim.frame_count += 1
        if self.anim.frame_count >= self.anim.frame_speed:
            self.anim.current_frame = (self.anim.current_frame + 1) % len(
                self.jugador_frames
            )
            self.anim.frame_count = 0

    def _update_background(self) -> None:
        self.render.fondo_x1 -= self.fondo_speed
        if self.render.fondo_x1 <= -self.w:
            self.render.fondo_x1 += self.w
        self.render.fondo_x2 = self.render.fondo_x1 + self.w

    def _update_bullet(self) -> None:
        if self.bullet.disparada:
            self.bullet.dist_min = mover_bala(
                self.bala, self.bullet.velocidad, self.jugador.x, self.bullet.dist_min
            )
        if self.bala.x < -self.bullet_size[0]:
            self.bullet.disparada, self.bullet.arriba, self.bullet.dist_min = reset_bala(
                self.bala, self.w - self.margin_x - self.bullet_size[0]
            )

    def _update_player(self) -> None:
        if self.player.salto:
            salto, salto_vel, en_suelo = aplicar_salto(
                self.jugador,
                self.player.salto,
                self.player.salto_vel,
                self.gravedad,
                self.ground_y,
            )
            self.player.salto = salto
            self.player.salto_vel = salto_vel if salto else self.salto_vel_inicial
            self.player.en_suelo = en_suelo

        agachado, timer = actualizar_agacharse(
            self.player.agachado,
            self.player.agachado_timer,
        )
        self.player.agachado = agachado
        self.player.agachado_timer = timer

    def _render(self) -> None:
        dibujar_fondo(
            self.pantalla,
            self.fondo_img,
            self.render.fondo_x1,
            self.render.fondo_x2,
        )

        self._update_animation()
        if self.player.salto:
            self.pantalla.blit(self.jugador_jump, (self.jugador.x, self.jugador.y))
        elif self.player.agachado:
            self.pantalla.blit(self.jugador_down, (self.jugador.x, self.jugador.y))
        else:
            dibujar_personaje(
                self.pantalla,
                self.jugador_frames,
                self.anim.current_frame,
                self.jugador.x,
                self.jugador.y,
            )
        dibujar_nave(self.pantalla, self.nave_img, self.nave.x, self.nave.y)

        bala_img = self.bala_img_arriba if self.bullet.arriba else self.bala_img_abajo
        dibujar_bala(self.pantalla, bala_img, self.bala.x, self.bala.y)

        if (self.model.entrenado_mlp or self.model.entrenado_arbol) and self.modo_auto:
            dibujar_info_modelo(
                self.pantalla, self.fuente_chica, self.model.ultima_proba
            )
            accion_txt = f"accion={self.model.accion_auto}"
            accion_render = self.fuente_chica.render(
                accion_txt, True, (255, 220, 120)
            )
            self.pantalla.blit(accion_render, (10, 30))

        if self.mostrar_hitbox:
            pygame.draw.rect(self.pantalla, (80, 220, 80), self.jugador, 2)
            pygame.draw.rect(self.pantalla, (240, 80, 80), self.bala, 1)
            pygame.draw.rect(self.pantalla, (240, 80, 80), self._calcular_hitbox_bala(), 2)

    def _check_collision(self) -> None:
        bala_hitbox = self._calcular_hitbox_bala()
        if self.jugador.colliderect(bala_hitbox):
            self._reset_estado_juego()

    def _update_frame(self) -> None:
        self._update_background()
        self._update_bullet()
        self._update_player()
        self._check_collision()
        self._render()

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()
        self._musica_game()

        while self.corriendo:
            procesar_eventos(
                pygame.event.get(),
                self.modo_auto,
                self.player.en_suelo,
                on_quit=lambda: setattr(self, "corriendo", False),
                on_menu=self._accion_volver_menu,
                on_fullscreen=self._toggle_fullscreen,
                on_resize=self._on_resize,
                on_jump=self._accion_salto_manual,
                on_crouch_press=self._accion_agacharse_manual,
            )

            if not self.corriendo:
                break

            if self.modo_auto:
                accion, _ = self._decidir_accion_auto()
                if accion == 1:
                    self.player.salto = iniciar_salto(
                        self.jugador, self.player.en_suelo, self.player.agachado
                    )
                elif accion == 2:
                    self._iniciar_impulso_agacharse_auto()
            else:
                self.registrar_decision_manual()

            if not self.bullet.disparada:
                self.bullet.disparada, self.bullet.velocidad, self.bullet.arriba = disparar_bala(
                    self.bala,
                    self.bullet.disparada,
                    self.bullet.velocidad,
                    self.scale,
                    self.bullet.arriba,
                    self.ground_y,
                    self.player_size[1],
                    self.bullet_size[1],
                )

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()

    def _accion_salto_manual(self) -> None:
        self.player.salto = iniciar_salto(
            self.jugador, self.player.en_suelo, self.player.agachado
        )
        if self.player.salto:
            self._accion_manual = 1

    def _accion_volver_menu(self) -> None:
        self._reset_estado_juego()
        self.mostrar_menu()

    def _accion_agacharse_manual(self) -> None:
        if self.player.agachado_timer > 0 or not self.player.en_suelo:
            return
        agachado, timer = iniciar_agacharse(self.player.en_suelo)
        if agachado:
            self.player.agachado = True
            self.player.agachado_timer = timer
            self._accion_manual = 2

    def _iniciar_impulso_agacharse_auto(self) -> None:
        if self.player.agachado_timer > 0:
            return
        agachado, timer = iniciar_agacharse(self.player.en_suelo)
        if agachado:
            self.player.agachado = True
            self.player.agachado_timer = timer
