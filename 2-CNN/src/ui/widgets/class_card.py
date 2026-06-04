from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


CLASS_EMOJI = {
    "ranas": "🐸",
    "aranas": "🕷️",
    "monos": "🐒",
    "pajaros": "🐦",
    "ballenas": "🐋",
    "changos": "🐒",
    "simios": "🐒",
}


def emoji_for(class_name: str) -> str:
    return CLASS_EMOJI.get(class_name, "🐾")


class ClassCard(QFrame):
    def __init__(self, class_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.class_name = class_name

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(8)

        self._emoji = QLabel(emoji_for(class_name))
        self._emoji.setObjectName("class_emoji")
        self._emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._name = QLabel(class_name.capitalize())
        self._name.setObjectName("class_name")

        self._count = QLabel("—")
        self._count.setObjectName("class_count")
        self._count.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        root.addWidget(self._emoji)
        root.addWidget(self._name, stretch=1)
        root.addWidget(self._count)

    def set_count(self, count: int) -> None:
        self._count.setText(f"{count:,}".replace(",", "."))


class ClassListPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("CLASES")
        title.setObjectName("section")
        layout.addWidget(title)

        self._cards: dict[str, ClassCard] = {}
        self._total_label = QLabel("Total: —")
        self._total_label.setObjectName("class_count_strong")
        self._total_label.setAlignment(Qt.AlignmentFlag.AlignRight)

    def set_classes(self, class_names: list[str]) -> None:
        old = self._cards
        parent_layout = self.layout()
        while parent_layout.count() > 1:
            item = parent_layout.takeAt(1)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for name, card in old.items():
            card.setParent(None)
            card.deleteLater()
        self._cards = {}

        for name in class_names:
            card = ClassCard(name)
            self._cards[name] = card
            parent_layout.addWidget(card)

        parent_layout.addStretch(1)
        parent_layout.addWidget(self._total_label)

    def set_counts(self, counts: dict[str, int]) -> None:
        total = 0
        for name, card in self._cards.items():
            n = counts.get(name, 0)
            card.set_count(n)
            total += n
        self._total_label.setText(f"Total: {total:,}".replace(",", "."))
