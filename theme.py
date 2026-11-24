"""Dark theme tokens and Qt Style Sheet for Qwen3-TTS.

Token values come from docs/DESIGN.md section 4 (proposed, not an approved brand
system). Styling is applied once at the application level by `apply_theme`; widgets
opt into variants with `set_role` (a dynamic `role` property) instead of inline QSS.

Roles
  QPushButton: "primary" (one per screen), "quiet" (no chrome), "nav" (sidebar item),
               "chip" (checkable suggestion), "segment" (two-way source switch), "info" (the (i) beside a label)
  QLabel:      "title", "subtitle", "heading", "field", "caption", "muted", "pill",
               "brand", "error-title", "notice"
  QFrame:      "divider", "callout"; QListWidget: "voices" (My Voices rows); object names "sidebar", "footer", "previewPanel"
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
    # Primary action (Generate voice): a deeper orange so white text reads clearly.
    "primary": "#C5540A",
    "primary_hover": "#B24C08",
    "primary_pressed": "#A94708",
    "primary_pressed_edge": "#7E3405",
    "primary_disabled": "#6B4A38",
    "on_primary": "#FFFFFF",
    "on_primary_disabled": "#D9C9BF",  # softened white; still readable on primary_disabled
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
QLabel#infoPopup {{ background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border_control']}; border-radius: 6px; padding: 8px 10px; }}
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
    background: {c['primary']}; color: {c['on_primary']}; border: 1px solid {c['primary']};
    font-weight: 600; padding: 10px 24px;
}}
QPushButton[role="primary"]:hover {{ background: {c['primary_hover']}; border-color: {c['primary_hover']}; }}
QPushButton[role="primary"]:pressed {{ background: {c['primary_pressed']}; border-color: {c['primary_pressed_edge']}; }}
QPushButton[role="primary"]:focus {{ border: 2px solid {c['text']}; padding: 9px 23px; outline: none; }}
QPushButton[role="primary"]:disabled {{
    background: {c['primary_disabled']}; color: {c['on_primary_disabled']}; border: 1px solid {c['primary_disabled']};
}}

QPushButton[role="quiet"] {{
    background: transparent; border: 1px solid transparent; color: {c['text_secondary']}; padding: 7px 10px;
}}
QPushButton[role="quiet"]:hover {{ background: {c['surface_hover']}; color: {c['text']}; }}
QPushButton[role="quiet"]:focus {{ border: 2px solid {c['accent']}; padding: 6px 9px; }}
QPushButton[role="quiet"]:disabled {{ background: transparent; border-color: transparent; color: {c['text_disabled']}; }}

QPushButton[role="info"] {{
    background: transparent; border: 2px solid transparent; border-radius: 11px; padding: 0;
    min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px;   /* + 2px border = the 22 px target */
}}
QPushButton[role="info"]:hover {{ background: {c['surface_hover']}; }}
QPushButton[role="info"]:focus {{ border: 2px solid {c['accent']}; padding: 0; }}

QPushButton[role="nav"] {{
    background: transparent; border: 1px solid transparent; color: {c['text_secondary']};
    text-align: left; padding: 8px 12px;
}}
QPushButton[role="nav"]:hover {{ background: {c['nav_hover']}; color: {c['text']}; }}
QPushButton[role="nav"]:checked {{ background: {c['nav_selected']}; color: {c['text']}; font-weight: 600; }}
QPushButton[role="nav"]:focus {{ border: 2px solid {c['accent']}; padding: 7px 11px; }}

QPushButton[role="segment"] {{
  background: {c['surface']}; color: {c['text_secondary']}; border: 1px solid {c['border']};
  padding: 8px 16px; min-height: 20px;
}}
QPushButton[role="segment"][position="left"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
QPushButton[role="segment"][position="right"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; border-left: none; }}
QPushButton[role="segment"]:hover {{ background: {c['surface_hover']}; color: {c['text']}; }}
QPushButton[role="segment"]:checked {{ background: {c['surface_hover']}; color: {c['text']}; font-weight: 600; }}
QPushButton[role="segment"]:focus {{ border: 2px solid {c['accent']}; padding: 7px 15px; }}
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
/* numeric fields without arrow buttons (role "numeric"): the field's own padding carries the height, so the
   inner line edit must not add a second border and padding; keyboard arrows and typing are unchanged */
QAbstractSpinBox[role="numeric"] {{ min-height: 20px; max-height: 20px; }}
QAbstractSpinBox[role="numeric"]::up-button, QAbstractSpinBox[role="numeric"]::down-button {{ width: 0; border: none; }}
QAbstractSpinBox[role="numeric"] QLineEdit {{ background: transparent; border: none; padding: 0; min-height: 0; }}
QPlainTextEdit[role="reader"], QTextEdit[role="reader"] {{
    background: transparent; border: none; border-left: 2px solid {c['border']}; border-radius: 0;
    padding: 2px 8px 2px 12px; color: {c['text']};
}}
QPlainTextEdit[role="reader"]:focus, QTextEdit[role="reader"]:focus {{
    border: 2px solid {c['accent']}; border-radius: 4px; padding: 0px 6px 0px 10px;
}}
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

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c['text_disabled']}; border-radius: 3px; min-height: 28px; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background: {c['text_disabled']}; border-radius: 3px; min-width: 28px; margin: 2px 0; }}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {c['text_muted']}; }}
QScrollBar::handle:vertical:pressed, QScrollBar::handle:horizontal:pressed {{ background: {c['text_secondary']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
    background: {c['text']}; border: 2px solid {c['text']}; }}
QSlider::handle:horizontal:hover {{ background: {c['accent_hover']}; border-color: {c['accent_hover']}; }}
QSlider::handle:horizontal:focus {{ border: 2px solid {c['accent']}; background: {c['text']}; margin: -6px 0;
    width: 16px; height: 16px; border-radius: 8px; }}
QSlider::groove:horizontal:disabled {{ background: {c['border']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {c['text_disabled']}; }}
QSlider::handle:horizontal:disabled {{ background: {c['text_disabled']}; border-color: {c['text_disabled']}; }}
QProgressBar {{ background: {c['surface']}; border: none; border-radius: 2px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 2px; }}

QScrollArea {{ background: transparent; border: none; }}

QTableWidget {{
    background: {c['surface']}; color: {c['text']}; gridline-color: {c['border']};
    border: 1px solid {c['border']}; border-radius: {r}px;
}}
QListWidget[role="voices"] {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: {r}px; outline: none;
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
QFrame[role="callout"] {{ background: {c['canvas']}; border: 1px solid {c['border']}; border-radius: {r}px; }}
QFrame[role="callout"] QLabel {{ background: transparent; border: none; }}
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


def style_numeric(spin, width=110, name=None):
    """The shared look for numeric fields: same height, border and radius as the other fields, no stepper
    buttons drawn. Range, step, default, typing, Up/Down keys and the wheel are untouched. `width` is a fixed
    pixel width, "fit" (the widest value the field can hold plus its padding, for crowded rows) or None
    (the layout decides)."""
    set_role(spin, "numeric")
    spin.lineEdit().setTextMargins(10, 0, 0, 0)   # the editor ignores the field padding when the stepper is zero-width
    if width == "fit":
        widest = max((spin.textFromValue(v) for v in (spin.minimum(), spin.maximum())), key=len)
        text = spin.prefix() + widest + spin.suffix()
        spin.setFixedWidth(max(64, spin.fontMetrics().horizontalAdvance(text) + 10 + 14))
    elif width:
        spin.setFixedWidth(width)
    if name:
        spin.setAccessibleName(name)
    return spin


def apply_theme(app):
    """Apply the dark palette, system font at body size, and the stylesheet to `app`."""
    app.setStyle("Fusion")  # native macOS style ignores much of QSS and the palette
    app.setPalette(_palette())
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(FONT_PX["body"])
    app.setFont(font)
    app.setStyleSheet(build_qss(arrow=chevron_icon()))
