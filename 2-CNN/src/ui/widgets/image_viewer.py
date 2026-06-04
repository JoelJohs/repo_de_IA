from __future__ import annotations

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class ImageViewer(QFrame):
    PLACEHOLDER = "Sin imagen\n\nSelecciona una imagen de la lista de la derecha"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._current_path: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._label = QLabel(self.PLACEHOLDER)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #565f89; font-size: 11pt;")
        self._label.setMinimumSize(QSize(280, 280))
        layout.addWidget(self._label)

    def clear(self) -> None:
        self._current_path = None
        self._label.clear()
        self._label.setText(self.PLACEHOLDER)
        self._label.setStyleSheet("color: #565f89; font-size: 11pt;")

    def set_image(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            self.clear()
            return
        self._current_path = path
        self._label.setStyleSheet("")
        scaled = pix.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._current_path is None:
            return
        pix = QPixmap(self._current_path)
        if pix.isNull():
            return
        scaled = pix.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def current_path(self) -> str | None:
        return self._current_path
