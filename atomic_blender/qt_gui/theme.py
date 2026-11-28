"""Commercial Obsidian/Teal QSS theme for Atomic Blender Studio (PySide6)."""

from __future__ import annotations

from typing import Any

PALETTE = {
    "bg": "#111419",
    "panel": "#181C23",
    "surface": "#222832",
    "surface_hover": "#2A3240",
    "border": "#2D3542",
    "accent": "#14B8A6",
    "accent_hover": "#0D9488",
    "accent_pressed": "#0F766E",
    "text": "#E6EDF3",
    "text_muted": "#94A3B8",
    "danger": "#EF4444",
    "danger_hover": "#DC2626",
}

DARK_STYLESHEET = f"""
QMainWindow, QDialog {{
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text"]};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 9pt;
}}

QWidget {{
    color: {PALETTE["text"]};
}}

QFrame#SidebarPanel {{
    background-color: {PALETTE["panel"]};
    border-right: 1px solid {PALETTE["border"]};
}}

QFrame#ToolbarFrame {{
    background-color: {PALETTE["panel"]};
    border-bottom: 1px solid {PALETTE["border"]};
}}

QFrame#CardGroup {{
    background-color: {PALETTE["bg"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
}}

QLabel#SectionHeader {{
    color: {PALETTE["text_muted"]};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.6px;
}}

QLabel#MutedLabel {{
    color: {PALETTE["text_muted"]};
    font-size: 8.5pt;
}}

QLabel#MeasureLabel {{
    color: #2DD4BF;
    font-weight: 600;
    font-size: 8.5pt;
}}

QTabWidget::pane {{
    border: 1px solid {PALETTE["border"]};
    border-radius: 6px;
    background-color: {PALETTE["panel"]};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text_muted"]};
    border: 1px solid {PALETTE["border"]};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 14px;
    margin-right: 2px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    background-color: {PALETTE["panel"]};
    color: {PALETTE["text"]};
    border-bottom: 2px solid {PALETTE["accent"]};
}}

QTabBar::tab:hover:!selected {{
    background-color: {PALETTE["surface_hover"]};
    color: {PALETTE["text"]};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {PALETTE["accent"]};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {PALETTE["accent"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    selection-background-color: {PALETTE["accent_pressed"]};
    selection-color: #FFFFFF;
}}

QPushButton, QToolButton {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
    padding: 4px 10px;
    font-weight: 600;
}}

QPushButton:hover, QToolButton:hover {{
    background-color: {PALETTE["surface_hover"]};
    border-color: #3E4859;
}}

QPushButton:pressed, QToolButton:pressed {{
    background-color: #1B2029;
}}

QPushButton:disabled, QToolButton:disabled {{
    color: #526074;
    background-color: #161A22;
    border-color: #222832;
}}

QPushButton[primary="true"] {{
    background-color: {PALETTE["accent"]};
    color: #062825;
    border: 1px solid #2DD4BF;
    font-weight: 700;
    padding: 7px 12px;
}}

QPushButton[primary="true"]:hover {{
    background-color: #2DD4BF;
}}

QPushButton[primary="true"]:pressed {{
    background-color: {PALETTE["accent_hover"]};
}}

QPushButton[primary="true"]:disabled {{
    background-color: #134E4A;
    color: #64748B;
    border-color: #115E59;
}}

QPushButton[danger="true"] {{
    background-color: #3B181C;
    color: #FCA5A5;
    border: 1px solid #7F1D1D;
}}

QPushButton[danger="true"]:hover {{
    background-color: {PALETTE["danger_hover"]};
    color: #FFFFFF;
}}

QPushButton[pill="true"] {{
    background-color: {PALETTE["panel"]};
    color: {PALETTE["text_muted"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 13px;
    padding: 5px 14px;
    font-weight: 600;
}}

QPushButton[pill="true"]:hover {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border-color: {PALETTE["accent"]};
}}

QPushButton[preset="true"] {{
    padding: 3px 6px;
    min-width: 28px;
}}

QCheckBox {{
    spacing: 6px;
    color: {PALETTE["text"]};
}}

QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid #475569;
    background-color: {PALETTE["surface"]};
}}

QCheckBox::indicator:checked {{
    background-color: {PALETTE["accent"]};
    border-color: #2DD4BF;
}}

QSlider::groove:horizontal {{
    border: none;
    height: 4px;
    background: {PALETTE["border"]};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {PALETTE["accent"]};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: #F8FAFC;
    border: 2px solid {PALETTE["accent"]};
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
}}

QListWidget, QPlainTextEdit {{
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 5px;
}}

QListWidget::item {{
    padding: 2px 6px;
}}

QListWidget::item:selected {{
    background-color: #134E4A;
    color: #5EEAD4;
}}

QProgressBar {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 4px;
    text-align: center;
    min-height: 10px;
    max-height: 12px;
}}

QProgressBar::chunk {{
    background-color: {PALETTE["accent"]};
    border-radius: 3px;
}}

QSplitter::handle {{
    background-color: {PALETTE["border"]};
}}

QStatusBar {{
    background-color: {PALETTE["panel"]};
    color: {PALETTE["text_muted"]};
    border-top: 1px solid {PALETTE["border"]};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background: {PALETTE["panel"]};
    width: 9px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {PALETTE["border"]};
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #475569;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QToolTip {{
    background-color: #0F172A;
    color: #E2E8F0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 4px 8px;
}}

QMenu {{
    background-color: {PALETTE["panel"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["border"]};
}}

QMenu::item:selected {{
    background-color: {PALETTE["accent_pressed"]};
    color: #FFFFFF;
}}
"""


def apply_studio_theme(app: Any) -> None:
    """Apply the commercial dark Fusion theme and stylesheet to ``QApplication``."""
    try:
        from PySide6.QtGui import QColor, QPalette
    except Exception:
        return

    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(PALETTE["bg"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(PALETTE["text"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(PALETTE["surface"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PALETTE["panel"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#0F172A"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#E2E8F0"))
    pal.setColor(QPalette.ColorRole.Text, QColor(PALETTE["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(PALETTE["surface"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(PALETTE["text"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(PALETTE["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#062825"))
    app.setPalette(pal)
    app.setStyleSheet(DARK_STYLESHEET)
