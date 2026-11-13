"""Dark theme tokens and Qt Style Sheet for Qwen3-TTS.

Token values come from docs/DESIGN.md section 4 (proposed, not an approved brand
system). Styling is applied once at the application level by `apply_theme`; widgets
opt into variants with `set_role` (a dynamic `role` property) instead of inline QSS.

Roles
  QPushButton: "primary" (one per screen), "quiet" (no chrome), "nav" (sidebar item),
               "chip" (checkable suggestion)
  QLabel:      "title", "subtitle", "heading", "field", "caption", "muted", "pill",
               "brand", "error-title", "notice"
  QFrame:      "divider"; object names "sidebar", "footer", "previewPanel"
"""

import os
import tempfile

from PySide6 import QtCore, QtGui, QtWidgets

COLORS = {
    "canvas": "#101114",
    "sidebar": "#15171B",
    "surface": "#1B1D22",
    "surface_hover": "#252831",
    "nav_hover": "#1E2026",       # sidebar hover: visible on the sidebar, lighter than nothing, below selected
    "nav_selected": "#2B2F39",    # neutral selected navigation surface
    "border": "#383B43",          # separators, decorative outlines
    "border_control": "#7A808C",  # input/button edges: >= 3:1 on canvas (WCAG 1.4.11)
    "text": "#F5F5F5",
    "text_secondary": "#B7BBC4",
    "text_muted": "#9399A4",
    "text_disabled": "#6B707B",
    "accent": "#F25C07",
    "accent_hover": "#FF7A2E",
    "on_accent": "#101114",       # white on the accent is only 3.3:1
    "selection": "#3A3F4B",       # neutral selection, not an orange fill
    "success": "#46BA82",
    "error": "#F07878",
}
SPACING = (4, 8, 12, 16, 24, 32)
RADII = {"control": 6, "surface": 10}
FONT_PX = {"body": 14, "secondary": 13, "title": 28, "heading": 17}
SIDEBAR_WIDTH = 208


def contrast_ratio(fg, bg):
    """WCAG 2.x contrast ratio between two '#RRGGBB' colors."""
    def lum(color):
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def chevron_icon(color=COLORS["text_secondary"]):
    """Draw the combo-box chevron once per run and return its file path (QSS needs a file)."""
    folder = os.path.join(tempfile.gettempdir(), "qwen3-tts-theme")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "chevron-down.png")
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.8, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    painter.drawPolyline([QtCore.QPointF(2.5, 4.5), QtCore.QPointF(6, 8), QtCore.QPointF(9.5, 4.5)])
    painter.end()
    if not pixmap.save(path) or not os.path.isfile(path):
        return None
    return path.replace("\\", "/")


def build_qss(c=COLORS, arrow=None):
    r = RADII["control"]
    s = RADII["surface"]
    # Flatten the native drop-down only when our chevron exists; otherwise keep the native one.
    arrow_rule = (f'QComboBox::drop-down {{ border: none; background: transparent; width: 28px; }}\n'
                  f'QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 12px; }}\n'
                  f'QPushButton[role="menu"]::menu-indicator {{ image: url("{arrow}"); width: 12px; height: 12px;'
                  f' subcontrol-origin: padding; subcontrol-position: right center; right: 10px; }}'
                  if arrow else "")
    f = FONT_PX
    return f"""
QToolTip {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px 6px; }}

/* ---- buttons: 36 px standard, 42 px primary */
QPushButton {{
    background: {c['surface']}; color: {c['text']};
    border: 1px solid {c['border_control']}; border-radius: {r}px;
    padding: 7px 14px; min-height: 20px;
}}
QPushButton:hover {{ background: {c['surface_hover']}; }}
QPushButton:pressed {{ background: {c['canvas']}; }}
QPushButton:focus {{ border: 2px solid {c['accent']}; padding: 6px 13px; }}
QPushButton:disabled {{ color: {c['text_disabled']}; background: {c['sidebar']}; border-color: {c['border']}; }}
QPushButton[role="menu"] {{ padding-right: 38px; }}
QPushButton[role="menu"]:focus {{ padding-right: 37px; }}
QMenu {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border_control']}; padding: 4px; }}
QMenu::item {{ padding: 7px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['nav_selected']}; color: {c['text']}; }}
QMenu::item:disabled {{ color: {c['text_disabled']}; }}

QPushButton[role="primary"] {{
    background: {c['accent']}; color: {c['on_accent']}; border: 1px solid {c['accent']};
    font-weight: 600; padding: 10px 24px;
}}
QPushButton[role="primary"]:hover {{ background: {c['accent_hover']}; border-color: {c['accent_hover']}; }}
QPushButton[role="primary"]:pressed {{ background: {c['accent']}; }}
QPushButton[role="primary"]:focus {{ border: 2px solid {c['text']}; padding: 9px 23px; }}
QPushButton[role="primary"]:disabled {{
    background: {c['sidebar']}; color: {c['text_disabled']}; border: 1px solid {c['border']};
}}

QPushButton[role="quiet"] {{
    background: transparent; border: 1px solid transparent; color: {c['text_secondary']}; padding: 7px 10px;
}}
QPushButton[role="quiet"]:hover {{ background: {c['surface_hover']}; color: {c['text']}; }}
QPushButton[role="quiet"]:focus {{ border: 2px solid {c['accent']}; padding: 6px 9px; }}
QPushButton[role="quiet"]:disabled {{ background: transparent; border-color: transparent; color: {c['text_disabled']}; }}

QPushButton[role="nav"] {{
    background: transparent; border: 1px solid transparent; color: {c['text_secondary']};
    text-align: left; padding: 8px 12px;
}}
QPushButton[role="nav"]:hover {{ background: {c['nav_hover']}; color: {c['text']}; }}
QPushButton[role="nav"]:checked {{ background: {c['nav_selected']}; color: {c['text']}; font-weight: 600; }}
QPushButton[role="nav"]:focus {{ border: 2px solid {c['accent']}; padding: 7px 11px; }}

QPushButton[role="chip"] {{
    border: 1px solid {c['border_control']}; border-radius: 15px; padding: 4px 12px; min-height: 20px;
    background: transparent; color: {c['text_secondary']};
}}
QPushButton[role="chip"]:hover {{ background: {c['surface_hover']}; color: {c['text']}; }}
QPushButton[role="chip"]:checked {{
    background: {c['selection']}; color: {c['text']}; border-color: {c['text']};
}}
QPushButton[role="chip"]:focus {{ border: 2px solid {c['accent']}; padding: 3px 11px; }}

QTableWidget QPushButton {{ padding: 0 4px; min-height: 0; }}

/* ---- inputs: 36 px single-line, neutral border, accent focus */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: 20px; padding: 7px 10px; }}
QPlainTextEdit, QTextEdit {{ padding: 8px 10px; }}
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c['surface']}; color: {c['text']};
    border: 1px solid {c['border_control']}; border-radius: {r}px;
    selection-background-color: {c['selection']}; selection-color: {c['text']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {c['accent']}; padding: 6px 9px;
}}
QPlainTextEdit:focus, QTextEdit:focus {{ border: 2px solid {c['accent']}; padding: 7px 9px; }}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QSpinBox:disabled,
QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {c['sidebar']}; color: {c['text_disabled']}; border-color: {c['border']};
}}
QPlainTextEdit[readOnly="true"], QTextEdit[readOnly="true"] {{ background: {c['sidebar']}; color: {c['text_secondary']}; }}
{arrow_rule}
QComboBox QAbstractItemView {{
    background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border_control']};
    selection-background-color: {c['selection']}; selection-color: {c['text']}; outline: 0;
}}
QTableWidget QComboBox, QTableWidget QLineEdit, QTableWidget QSpinBox {{ padding: 2px 8px; min-height: 0; }}

QCheckBox {{ border: 1px solid transparent; border-radius: 4px; padding: 1px 3px; }}
QCheckBox:focus {{ border: 1px solid {c['accent']}; }}
QCheckBox::indicator:unchecked {{
    width: 14px; height: 14px; background: {c['surface']};
    border: 1px solid {c['border_control']}; border-radius: 3px;
}}

QGroupBox {{ border: 1px solid {c['border']}; border-radius: {s}px; margin-top: 12px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['text_secondary']}; }}

QProgressBar {{ background: {c['surface']}; border: none; border-radius: 2px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 2px; }}

QScrollArea {{ background: transparent; border: none; }}

QTableWidget {{
    background: {c['surface']}; color: {c['text']}; gridline-color: {c['border']};
    border: 1px solid {c['border']}; border-radius: {r}px;
}}
QTableWidget::item {{ padding: 0 4px; }}
QTableWidget::item:selected {{ background: {c['selection']}; color: {c['text']}; }}
QHeaderView::section {{
    background: {c['canvas']}; color: {c['text_secondary']}; border: none;
    border-bottom: 1px solid {c['border']}; padding: 6px 8px;
}}

/* ---- compact density: pages that keep their original dense layout (Clone, My Voices, Audiobooks) */
QWidget[density="compact"] QPushButton {{ padding: 4px 12px; }}
QWidget[density="compact"] QPushButton:focus {{ padding: 3px 11px; }}
QWidget[density="compact"] QPushButton[role="menu"] {{ padding-right: 38px; }}
QWidget[density="compact"] QPushButton[role="menu"]:focus {{ padding-right: 37px; }}
QWidget[density="compact"] QPushButton[role="primary"] {{ padding: 8px 20px; }}
QWidget[density="compact"] QPushButton[role="primary"]:focus {{ padding: 7px 19px; }}
QWidget[density="compact"] QTableWidget QPushButton {{ padding: 0 4px; min-height: 0; }}
QWidget[density="compact"] QLineEdit, QWidget[density="compact"] QSpinBox,
QWidget[density="compact"] QDoubleSpinBox, QWidget[density="compact"] QComboBox {{ padding: 4px 8px; }}
QWidget[density="compact"] QLineEdit:focus, QWidget[density="compact"] QSpinBox:focus,
QWidget[density="compact"] QDoubleSpinBox:focus, QWidget[density="compact"] QComboBox:focus {{ padding: 3px 7px; }}
QWidget[density="compact"] QTableWidget QComboBox, QWidget[density="compact"] QTableWidget QLineEdit,
QWidget[density="compact"] QTableWidget QSpinBox {{ padding: 2px 8px; min-height: 0; }}

/* ---- structure */
QFrame#sidebar {{ background: {c['sidebar']}; border: none; border-right: 1px solid {c['border']}; }}
QFrame#footer {{ background: {c['canvas']}; border: none; border-top: 1px solid {c['border']}; }}
QFrame#previewPanel {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {s}px; }}
QFrame#previewPanel[empty="true"] {{ background: transparent; }}
QFrame[role="divider"] {{ background: {c['border']}; border: none; max-height: 1px; min-height: 1px; }}

/* ---- text roles */
QLabel[role="brand"] {{ color: {c['text']}; font-size: 15px; font-weight: 600; padding: 4px 9px; }}
QLabel[role="title"] {{ color: {c['text']}; font-size: {f['title']}px; font-weight: 600; }}
QLabel[role="subtitle"] {{ color: {c['text_secondary']}; font-size: {f['body']}px; }}
QLabel[role="heading"] {{ color: {c['text']}; font-size: {f['heading']}px; font-weight: 600; }}
QLabel[role="field"] {{ color: {c['text']}; font-weight: 600; }}
QLabel[role="caption"] {{ color: {c['text_muted']}; font-size: {f['secondary']}px; }}
QLabel[role="muted"] {{ color: {c['text_muted']}; font-size: {f['secondary']}px; }}
QLabel[role="notice"] {{ color: {c['text_secondary']}; font-size: {f['secondary']}px; }}
QLabel[role="error-title"] {{ color: {c['error']}; font-size: {f['body']}px; font-weight: 600; }}
QLabel[role="error-text"] {{ color: {c['error']}; font-size: {f['secondary']}px; }}
QLabel[role="pill"] {{
    background: {c['sidebar']}; border: 1px solid {c['border']}; border-radius: 9px;
    color: {c['text_secondary']}; font-size: {f['secondary']}px; padding: 2px 10px;
}}
"""


def _palette(c=COLORS):
    p = QtGui.QPalette()
    Q, G = QtGui.QPalette, QtGui.QPalette.ColorGroup
    roles = {
        Q.Window: c["canvas"], Q.WindowText: c["text"], Q.Base: c["surface"],
        Q.AlternateBase: c["sidebar"], Q.Text: c["text"], Q.Button: c["surface"],
        Q.ButtonText: c["text"], Q.ToolTipBase: c["surface"], Q.ToolTipText: c["text"],
        Q.Highlight: c["selection"], Q.HighlightedText: c["text"],
        Q.PlaceholderText: c["text_muted"], Q.Link: c["accent_hover"],
    }
    for role, color in roles.items():
        p.setColor(role, QtGui.QColor(color))
    for role in (Q.WindowText, Q.Text, Q.ButtonText):
        p.setColor(G.Disabled, role, QtGui.QColor(c["text_disabled"]))
    return p


def set_role(widget, role):
    """Tag a widget with a style variant (see module docstring) and refresh its style."""
    widget.setProperty("role", role)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def apply_theme(app):
    """Apply the dark palette, system font at body size, and the stylesheet to `app`."""
    app.setStyle("Fusion")  # native macOS style ignores much of QSS and the palette
    app.setPalette(_palette())
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(FONT_PX["body"])
    app.setFont(font)
    app.setStyleSheet(build_qss(arrow=chevron_icon()))
