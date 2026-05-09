import os
import random
from typing import List, Optional, Tuple

import pygame

from core.constantes import BASE_H, BASE_W, EXTRA_SCALE
from core.tipos import Sample
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

        self.datos_modelo: List[Sample] = []
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica: Optional[int] = None
        self.ultima_proba_salto: Optional[float] = None

        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (38, 58)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        self.fondo_speed = 3

        self.salto = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.velocidad_bala = -12
        self.bala_disparada = False
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin = int(50 * self.scale)
        ground_offset = int(100 * self.scale)
        self.ground_y = self.h - ground_offset

        self.player_size = (int(38 * self.scale), int(58 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

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
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0],
                self.ship_size[1],
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
        self.bala_img = assets["bala_img"]
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
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada = False
        self.velocidad_bala = int(-10 * self.scale)
        self.salto = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    def exportar_datos_csv(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return exportar_datos_csv(self.datos_modelo, base)

    def graficar_datos_2d(self) -> str:
        return graficar_datos_2d(self.datos_modelo)

    def graficar_datos_3d(self) -> str:
        return graficar_datos_3d(self.datos_modelo)

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo:
            self.salto = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    def registrar_decision_manual(self) -> None:
        registrar_decision_manual(
            self.datos_modelo,
            self.bala_disparada,
            self.jugador.x,
            self.bala.x,
            self.en_suelo,
            self.velocidad_bala,
        )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        modelo, scaler, clase_unica, mensaje = entrenar_modelo(self.datos_modelo)
        if modelo is None and clase_unica is None:
            return False, mensaje

        self._reset_modelo()
        self.modelo = modelo
        self.scaler = scaler
        self.modelo_entrenado = True
        self.clase_unica = clase_unica
        return True, mensaje

    def decision_auto_saltar(self) -> bool:
        if not self.modelo_entrenado:
            return False

        decision, proba_salto = decision_auto_saltar(
            self.modelo,
            self.scaler,
            self.clase_unica,
            self.bala_disparada,
            self.en_suelo,
            self.jugador.x,
            self.bala.x,
            self.velocidad_bala,
        )
        self.ultima_proba_salto = proba_salto
        return decision

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
                len(self.datos_modelo),
                self.modelo_entrenado,
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
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
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

    def _update_frame(self) -> None:
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        dibujar_fondo(self.pantalla, self.fondo_img, self.fondo_x1, self.fondo_x2)

        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0
        if self.salto:
            self.pantalla.blit(self.jugador_jump, (self.jugador.x, self.jugador.y))
        else:
            dibujar_personaje(
                self.pantalla,
                self.jugador_frames,
                self.current_frame,
                self.jugador.x,
                self.jugador.y,
            )
        dibujar_nave(self.pantalla, self.nave_img, self.nave.x, self.nave.y)

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        dibujar_bala(self.pantalla, self.bala_img, self.bala.x, self.bala.y)

        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        if self.modelo_entrenado and self.modo_auto:
            dibujar_info_modelo(
                self.pantalla, self.fuente_chica, self.ultima_proba_salto
            )

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif (
                        e.key == pygame.K_SPACE
                        and (not self.modo_auto)
                        and self.en_suelo
                    ):
                        salto_frame = True
                        self.iniciar_salto()

            if not self.corriendo:
                break

            if self.modo_auto:
                if self.decision_auto_saltar():
                    self.iniciar_salto()
            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()
