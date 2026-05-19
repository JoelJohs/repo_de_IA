from __future__ import annotations

import os
from collections import deque
from typing import Deque, Optional, Tuple

import pygame

from core.constantes import BASE_H, BASE_W, EXTRA_SCALE
from ml.dataset import exportar_datos_csv, registrar_decision_manual
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
    ScoreState,
)
from .fisica import aplicar_salto, iniciar_agacharse, iniciar_salto, terminar_agacharse
from .loop import procesar_eventos
from .puntaje import calcular_bonus_esquiva


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + MLP (solo memoria)")

        self.corriendo = True
        self.modo_auto = False

        self.model = ModelState(
            datos=[],
            modelo=None,
            scaler=None,
            entrenado=False,
            clase_unica=None,
            ultima_proba=None,
            accion_auto="none",
        )

        self._accion_manual = 0
        self._crouch_hold = False
        self._estilo_hist: Deque[int] = deque(maxlen=200)
        self._estilo_min_muestras = 40
        self._estilo_umbral = 0.55
        self._auto_crouch_hold_frames = 0
        self._auto_crouch_hold_max = 20
        self._auto_crouch_hold_infinite = False
        self._auto_crouch_cycle = False
        self._auto_crouch_on_remaining = 0
        self._auto_crouch_off_remaining = 0
        self._auto_crouch_on_len = 0
        self._auto_crouch_off_len = 0

        self._manual_prev_agachado = False
        self._manual_crouch_frames = 0
        self._manual_stand_frames = 0
        self._crouch_durations: Deque[int] = deque(maxlen=60)
        self._crouch_pauses: Deque[int] = deque(maxlen=60)

        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (60, 96)
        self.bullet_size = (64, 64)
        self.ship_size = (192, 192)
        self.fondo_speed = 3

        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0

        self.anim = AnimationState(current_frame=0, frame_speed=10, frame_count=0)
        self.render = RenderState(fondo_x1=0, fondo_x2=start_w)

        self._puntaje_frame = 1
        self._puntaje_esquiva_base = 25
        self._puntaje_esquiva_umbral = 200
        self.mostrar_hitbox = False
        self._hitbox_bala_scale = 0.4

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin = int(50 * self.scale)
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
        self._puntaje_esquiva_umbral = int(200 * self.scale)

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(
                self.margin, self.ground_y, self.player_size[0], self.player_size[1]
            )
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0],
                self.bullet_size[1],
            )
            nave_y = self.ground_y + self.player_size[1] - self.ship_size[1]
            nave_x = self.w - self.ship_size[0] - int(40 * self.scale)
            self.nave = pygame.Rect(nave_x, nave_y, self.ship_size[0], self.ship_size[1])

        self.player = PlayerState(
            rect=self.jugador,
            salto=False,
            en_suelo=True,
            salto_vel=self.salto_vel_inicial,
            agachado=False,
        )
        self.bullet = BulletState(
            rect=self.bala,
            velocidad=int(-10 * self.scale),
            disparada=False,
            arriba=False,
            dist_min=None,
        )
        self.score = ScoreState(
            valor=0,
            por_frame=self._puntaje_frame,
            esquiva_base=self._puntaje_esquiva_base,
            esquiva_umbral=self._puntaje_esquiva_umbral,
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
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.nave.x = self.w - self.ship_size[0] - int(40 * self.scale)
        self.nave.y = self.ground_y + self.player_size[1] - self.ship_size[1]
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)

        self.player.salto = False
        self.player.en_suelo = True
        self.player.salto_vel = self.salto_vel_inicial
        self.player.agachado = False

        self.bullet.disparada = False
        self.bullet.velocidad = int(-10 * self.scale)
        self.bullet.arriba = False
        self.bullet.dist_min = None

        self.score.valor = 0
        self.render.fondo_x1 = 0
        self.render.fondo_x2 = self.w
        self._decision_frame_counter = 0
        self._accion_manual = 0
        self._crouch_hold = False
        self._auto_crouch_hold_frames = 0
        self._auto_crouch_hold_infinite = False
        self._auto_crouch_cycle = False
        self._auto_crouch_on_remaining = 0
        self._auto_crouch_off_remaining = 0
        self._auto_crouch_on_len = 0
        self._auto_crouch_off_len = 0
        self._manual_prev_agachado = False
        self._manual_crouch_frames = 0
        self._manual_stand_frames = 0
        self._crouch_durations.clear()
        self._crouch_pauses.clear()

    def _reset_modelo(self) -> None:
        self.model.modelo = None
        self.model.scaler = None
        self.model.entrenado = False
        self.model.clase_unica = None
        self.model.ultima_proba = None
        self.model.accion_auto = "none"
        self._estilo_hist.clear()
        self._accion_manual = 0
        self._crouch_hold = False
        self._auto_crouch_hold_frames = 0
        self._auto_crouch_hold_infinite = False
        self._auto_crouch_cycle = False
        self._auto_crouch_on_remaining = 0
        self._auto_crouch_off_remaining = 0
        self._auto_crouch_on_len = 0
        self._auto_crouch_off_len = 0
        self._manual_prev_agachado = False
        self._manual_crouch_frames = 0
        self._manual_stand_frames = 0
        self._crouch_durations.clear()
        self._crouch_pauses.clear()

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
            if self.player.agachado or self._crouch_hold
            else 0
        )
        ataque_color = 1 if self.bullet.arriba else 0
        self._estilo_hist.append(accion)
        registrar_decision_manual(
            self.model.datos,
            self.bullet.disparada,
            self.jugador.x,
            self.bala.x,
            self.bala.y,
            self.player.en_suelo,
            self.player.agachado,
            accion,
            self.bullet.velocidad,
            self.score.valor,
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

    def _actualizar_patron_agacharse_manual(self) -> None:
        agachado = self.player.agachado or self._crouch_hold
        if agachado:
            self._manual_crouch_frames += 1
            if not self._manual_prev_agachado and self._manual_stand_frames > 0:
                self._crouch_pauses.append(self._manual_stand_frames)
                self._manual_stand_frames = 0
        else:
            self._manual_stand_frames += 1
            if self._manual_prev_agachado and self._manual_crouch_frames > 0:
                self._crouch_durations.append(self._manual_crouch_frames)
                self._manual_crouch_frames = 0
        self._manual_prev_agachado = agachado

    def _clamp(self, value: int, min_value: int, max_value: int) -> int:
        return max(min_value, min(max_value, value))

    def _calcular_patron_agacharse(self) -> Tuple[int, int]:
        base_on_min = int(6 * self.scale)
        base_on_max = int(45 * self.scale)
        base_off_min = int(6 * self.scale)
        base_off_max = int(60 * self.scale)
        if not self._crouch_durations:
            dur = int(14 * self.scale)
        else:
            dur = int(sum(self._crouch_durations) / len(self._crouch_durations))
        if not self._crouch_pauses:
            pausa = int(12 * self.scale)
        else:
            pausa = int(sum(self._crouch_pauses) / len(self._crouch_pauses))
        dur = self._clamp(dur, base_on_min, base_on_max)
        pausa = self._clamp(pausa, base_off_min, base_off_max)
        return dur, pausa

    def _aplicar_ciclo_agacharse_auto(self) -> int:
        if not self._auto_crouch_cycle:
            return 2
        if self._auto_crouch_on_remaining <= 0 and self._auto_crouch_off_remaining <= 0:
            self._auto_crouch_on_remaining = self._auto_crouch_on_len
        if self._auto_crouch_on_remaining > 0:
            self._auto_crouch_on_remaining -= 1
            if self._auto_crouch_on_remaining == 0:
                self._auto_crouch_off_remaining = self._auto_crouch_off_len
            return 2
        if self._auto_crouch_off_remaining > 0:
            self._auto_crouch_off_remaining -= 1
            if self._auto_crouch_off_remaining == 0:
                self._auto_crouch_on_remaining = self._auto_crouch_on_len
            return 0
        return 0

    def entrenar_modelo(self) -> Tuple[bool, str]:
        modelo, scaler, clase_unica, mensaje = entrenar_modelo(self.model.datos)
        if modelo is None and clase_unica is None:
            return False, mensaje

        self._reset_modelo()
        self.model.modelo = modelo
        self.model.scaler = scaler
        self.model.entrenado = True
        self.model.clase_unica = clase_unica
        return True, mensaje

    def _decidir_accion_auto(self) -> Tuple[int, Optional[float]]:
        if not self.model.entrenado:
            return 0, None

        accion, proba_salto = decision_auto_saltar(
            self.model.modelo,
            self.model.scaler,
            self.model.clase_unica,
            self.bullet.disparada,
            self.player.en_suelo,
            self.jugador.x,
            self.bala.x,
            self.bala.y,
            self.bullet.velocidad,
            self.score.valor,
            self.bullet.arriba,
        )
        self.model.ultima_proba = proba_salto
        accion_estilo = self._obtener_accion_estilo()
        if accion_estilo is not None:
            accion = accion_estilo
            if accion_estilo == 2:
                self._auto_crouch_hold_infinite = False
                self._auto_crouch_cycle = True
                on_len, off_len = self._calcular_patron_agacharse()
                if on_len != self._auto_crouch_on_len or off_len != self._auto_crouch_off_len:
                    self._auto_crouch_on_len = on_len
                    self._auto_crouch_off_len = off_len
                    self._auto_crouch_on_remaining = on_len
                    self._auto_crouch_off_remaining = 0
            else:
                self._auto_crouch_cycle = False
        else:
            self._auto_crouch_hold_infinite = False
            self._auto_crouch_cycle = False
            if accion == 2:
                self._auto_crouch_hold_frames = self._auto_crouch_hold_max
        self.model.accion_auto = (
            "salta" if accion == 1 else "agacha" if accion == 2 else "none"
        )
        return accion, proba_salto

    def mostrar_menu(self) -> None:
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
                self.model.entrenado,
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
                        if not self.model.entrenado:
                            msg = "Primero entrena el MLP (T) en esta sesión."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
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
        self.render.fondo_x2 -= self.fondo_speed
        if self.render.fondo_x1 <= -self.w:
            self.render.fondo_x1 = self.w
        if self.render.fondo_x2 <= -self.w:
            self.render.fondo_x2 = self.w

    def _update_bullet(self) -> None:
        if self.bullet.disparada:
            self.bullet.dist_min = mover_bala(
                self.bala, self.bullet.velocidad, self.jugador.x, self.bullet.dist_min
            )
        if self.bala.x < -self.bullet_size[0]:
            bonus = calcular_bonus_esquiva(
                self.bullet.dist_min,
                self.score.esquiva_umbral,
                self.score.esquiva_base,
            )
            if bonus:
                self.score.valor += bonus
            self.bullet.disparada, self.bullet.arriba, self.bullet.dist_min = reset_bala(
                self.bala, self.w - self.margin
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

        if self.model.entrenado and self.modo_auto:
            dibujar_info_modelo(
                self.pantalla, self.fuente_chica, self.model.ultima_proba
            )
            accion_txt = f"accion={self.model.accion_auto}"
            accion_render = self.fuente_chica.render(
                accion_txt, True, (255, 220, 120)
            )
            self.pantalla.blit(accion_render, (10, 30))

        score_txt = self.fuente_chica.render(
            f"puntaje={self.score.valor}", True, (255, 220, 120)
        )
        self.pantalla.blit(score_txt, (10, 50))

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

        while self.corriendo:
            procesar_eventos(
                pygame.event.get(),
                self.modo_auto,
                self.player.en_suelo,
                on_quit=lambda: setattr(self, "corriendo", False),
                on_menu=self._accion_volver_menu,
                on_fullscreen=self._toggle_fullscreen,
                on_jump=self._accion_salto_manual,
                on_crouch_start=self._accion_agacharse,
                on_crouch_end=self._accion_terminar_agacharse,
            )

            if not self.corriendo:
                break

            if self.modo_auto:
                accion, _ = self._decidir_accion_auto()
                if self._auto_crouch_cycle and accion == 2:
                    accion = self._aplicar_ciclo_agacharse_auto()
                if accion == 1:
                    self.player.salto = iniciar_salto(
                        self.jugador, self.player.en_suelo, self.player.agachado
                    )
                elif accion == 2:
                    self.player.agachado = iniciar_agacharse(
                        self.jugador, self.player.agachado, self.player.en_suelo, self.player_size[1]
                    )
                elif self.player.agachado:
                    if self._auto_crouch_hold_infinite:
                        self.player.agachado = iniciar_agacharse(
                            self.jugador,
                            self.player.agachado,
                            self.player.en_suelo,
                            self.player_size[1],
                        )
                    elif self._auto_crouch_hold_frames > 0:
                        self.player.agachado = iniciar_agacharse(
                            self.jugador,
                            self.player.agachado,
                            self.player.en_suelo,
                            self.player_size[1],
                        )
                        self._auto_crouch_hold_frames -= 1
                    else:
                        self.player.agachado = terminar_agacharse(
                            self.jugador, self.player.agachado, self.player_size[1]
                        )
            else:
                self._actualizar_patron_agacharse_manual()
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

            self.score.valor += self.score.por_frame

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

    def _accion_agacharse(self) -> None:
        self.player.agachado = iniciar_agacharse(
            self.jugador, self.player.agachado, self.player.en_suelo, self.player_size[1]
        )
        if self.player.agachado:
            self._crouch_hold = True

    def _accion_terminar_agacharse(self) -> None:
        self.player.agachado = terminar_agacharse(
            self.jugador, self.player.agachado, self.player_size[1]
        )
        if not self.player.agachado:
            self._crouch_hold = False
