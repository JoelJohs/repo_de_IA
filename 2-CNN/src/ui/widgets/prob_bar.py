from __future__ import annotations

from typing import List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget


class ProbRow(QWidget):
    def __init__(self, rank: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rank = rank

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self._label = QLabel("—")
        self._label.setObjectName("class_name")
        self._label.setMinimumWidth(90)
        self._label.setMaximumWidth(110)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(16)
        self._bar.setObjectName(f"top{min(rank, 3)}")
        self._bar.setMinimumWidth(120)

        self._pct = QLabel("0.00")
        self._pct.setObjectName("class_count")
        self._pct.setMinimumWidth(50)
        self._pct.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        layout.addWidget(self._label)
        layout.addWidget(self._bar, stretch=1)
        layout.addWidget(self._pct)

    def set_value(self, class_name: str, prob: float) -> None:
        self._label.setText(class_name.capitalize())
        self._bar.setValue(int(round(prob * 1000)))
        self._pct.setText(f"{prob:.4f}")


class ProbBarPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("TOP-K")
        title.setObjectName("section")
        layout.addWidget(title)

        self._rows: list[ProbRow] = []
        for i in range(1, 4):
            row = ProbRow(i)
            self._rows.append(row)
            layout.addWidget(row)
        layout.addStretch(1)

    def set_predictions(self, predictions: List[Tuple[str, float]]) -> None:
        for i, row in enumerate(self._rows):
            if i < len(predictions):
                name, prob = predictions[i]
                row.set_value(name, prob)
                row.setVisible(True)
            else:
                row.setVisible(False)
