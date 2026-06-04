from __future__ import annotations

from PyQt6.QtWidgets import QApplication

PALETTE = {
    "bg": "#1e1e2e",
    "panel": "#2a2a3a",
    "panel_alt": "#242433",
    "border": "#3a3a4a",
    "accent": "#7aa2f7",
    "accent_dim": "#5a7ec7",
    "ok": "#9ece6a",
    "err": "#f7768e",
    "text": "#c0caf5",
    "text_dim": "#9aa5ce",
    "text_muted": "#565f89",
    "warn": "#e0af68",
}

QSS = f"""
QMainWindow, QWidget#central {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
}}

QFrame#header, QFrame#panel, QFrame#card, QFrame#result {{
    background-color: {PALETTE['panel']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
}}

QFrame#header {{
    background-color: {PALETTE['panel_alt']};
}}

QFrame#card {{
    padding: 6px 10px;
}}

QFrame#result {{
    padding: 10px 14px;
}}

QLabel {{
    color: {PALETTE['text']};
    background: transparent;
}}

QLabel#title {{
    color: {PALETTE['accent']};
    font-size: 14pt;
    font-weight: 600;
}}

QLabel#subtitle {{
    color: {PALETTE['text_dim']};
    font-size: 9pt;
}}

QLabel#section {{
    color: {PALETTE['text_dim']};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 4px 0;
}}

QLabel#badge {{
    color: {PALETTE['ok']};
    background-color: {PALETTE['panel_alt']};
    border: 1px solid {PALETTE['ok']};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 9pt;
    font-weight: 600;
}}

QLabel#badge_warn {{
    color: {PALETTE['warn']};
    background-color: {PALETTE['panel_alt']};
    border: 1px solid {PALETTE['warn']};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 9pt;
    font-weight: 600;
}}

QLabel#class_emoji {{
    font-size: 18pt;
    padding-right: 6px;
}}

QLabel#class_name {{
    font-size: 11pt;
    font-weight: 500;
}}

QLabel#class_count {{
    color: {PALETTE['text_muted']};
    font-size: 10pt;
}}

QLabel#class_count_strong {{
    color: {PALETTE['accent']};
    font-size: 11pt;
    font-weight: 600;
}}

QLabel#real_label, QLabel#pred_label, QLabel#time_label {{
    color: {PALETTE['text_dim']};
    font-size: 10pt;
}}

QLabel#real_value, QLabel#pred_value {{
    color: {PALETTE['text']};
    font-size: 11pt;
    font-weight: 500;
}}

QLabel#ok {{
    color: {PALETTE['ok']};
    font-size: 11pt;
    font-weight: 600;
}}

QLabel#err {{
    color: {PALETTE['err']};
    font-size: 11pt;
    font-weight: 600;
}}

QListWidget {{
    background-color: {PALETTE['panel_alt']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 4px;
    color: {PALETTE['text']};
}}

QListWidget::item {{
    padding: 4px 8px;
    border-radius: 4px;
}}

QListWidget::item:selected {{
    background-color: {PALETTE['accent_dim']};
    color: white;
}}

QListWidget::item:hover {{
    background-color: {PALETTE['panel']};
}}

QPushButton {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 10pt;
}}

QPushButton:hover {{
    background-color: {PALETTE['accent_dim']};
    color: white;
    border-color: {PALETTE['accent']};
}}

QPushButton:pressed {{
    background-color: {PALETTE['accent']};
    color: {PALETTE['bg']};
}}

QPushButton:disabled {{
    color: {PALETTE['text_muted']};
    background-color: {PALETTE['panel']};
    border-color: {PALETTE['border']};
}}

QRadioButton {{
    color: {PALETTE['text_dim']};
    spacing: 6px;
    padding: 2px 4px;
}}

QRadioButton::indicator {{
    width: 12px;
    height: 12px;
    border-radius: 6px;
    border: 1px solid {PALETTE['border']};
    background-color: {PALETTE['panel_alt']};
}}

QRadioButton::indicator:checked {{
    background-color: {PALETTE['accent']};
    border-color: {PALETTE['accent']};
}}

QProgressBar {{
    background-color: {PALETTE['panel_alt']};
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    text-align: left;
    color: {PALETTE['text']};
    height: 16px;
}}

QProgressBar::chunk {{
    background-color: {PALETTE['accent']};
    border-radius: 3px;
}}

QProgressBar#top1::chunk {{
    background-color: {PALETTE['ok']};
}}

QProgressBar#top2::chunk {{
    background-color: {PALETTE['accent']};
}}

QProgressBar#top3::chunk {{
    background-color: {PALETTE['text_muted']};
}}

QStatusBar {{
    background-color: {PALETTE['panel_alt']};
    color: {PALETTE['text_muted']};
    border-top: 1px solid {PALETTE['border']};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background: {PALETTE['panel_alt']};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: {PALETTE['border']};
    border-radius: 4px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {PALETTE['text_muted']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {PALETTE['panel_alt']};
    height: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background: {PALETTE['border']};
    border-radius: 4px;
    min-width: 24px;
}}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
