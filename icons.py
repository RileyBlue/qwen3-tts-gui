"""Icon system: a small set of vendored Lucide SVGs turned into theme-coloured QIcons.

Qt Style Sheets cannot recolour an SVG (there is no ``currentColor`` support), so each icon is
rendered once per state with Qt's own SVG renderer, with the theme colour substituted into the
asset text, and handed to QIcon:

    Normal        resting
    Active        keyboard focus (Qt draws a focused button's icon in Active mode)
    Disabled      disabled control
    On (checked)  the selected navigation item, expanded disclosure, and so on

Pixmaps are rendered at 1x and 2x so they stay sharp on Retina and other high-DPI displays.
A missing or unreadable asset yields an empty QIcon, so the control simply shows its text label.
Icons never carry meaning alone: every control that uses one keeps its text and accessible name.

Assets live in ``assets/icons`` (Lucide, ISC licence, with the MIT Feather notice; see the LICENSE
file in that folder). Only icons the application uses are vendored.
"""

import functools
import os
import sys

from PySide6 import QtCore, QtGui, QtSvg

from theme import COLORS

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")

SIZE_COMPACT = 16   # buttons
SIZE_NAV = 18       # sidebar
GAP = 4             # transparent space after the glyph, so it never touches the button text

# Each role maps (mode, state) to a theme colour token. Neutral by default; only the primary
# action's icon takes the on-primary colour (white) so it matches its label.
_ROLES = {
    "default": {
        (QtGui.QIcon.Normal, QtGui.QIcon.Off): "text_secondary",
        (QtGui.QIcon.Active, QtGui.QIcon.Off): "text",
        (QtGui.QIcon.Disabled, QtGui.QIcon.Off): "text_disabled",
        (QtGui.QIcon.Normal, QtGui.QIcon.On): "text",
        (QtGui.QIcon.Active, QtGui.QIcon.On): "text",
        (QtGui.QIcon.Disabled, QtGui.QIcon.On): "text_disabled",
    },
    "primary": {
        (QtGui.QIcon.Normal, QtGui.QIcon.Off): "on_primary",
        (QtGui.QIcon.Active, QtGui.QIcon.Off): "on_primary",
        (QtGui.QIcon.Disabled, QtGui.QIcon.Off): "on_primary_disabled",
    },
}
_ROLES["nav"] = _ROLES["default"]
# Status glyphs always sit beside their text; the error one uses the theme's error colour.
_ROLES["error"] = {key: "error" for key in _ROLES["default"]}

_warned = set()


def _read(name):
    path = os.path.join(ICON_DIR, f"{name}.svg")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        if name not in _warned:
            _warned.add(name)
            print(f"icons: missing asset {path}", file=sys.stderr)
        return None


def _pixmap(svg_text, color, size, scale, gap):
    data = svg_text.replace("currentColor", color).encode("utf-8")
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(data))
    if not renderer.isValid():
        return None
    pixmap = QtGui.QPixmap(int((size + gap) * scale), int(size * scale))
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    renderer.render(painter, QtCore.QRectF(0, 0, size * scale, size * scale))
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


@functools.lru_cache(maxsize=None)
def icon(name, size=SIZE_COMPACT, role="default", gap=GAP):
    """Return a QIcon for a vendored Lucide icon, or an empty QIcon when it cannot be loaded."""
    result = QtGui.QIcon()
    svg_text = _read(name)
    if svg_text is None:
        return result
    for (mode, state), token in _ROLES.get(role, _ROLES["default"]).items():
        for scale in (1, 2):
            pixmap = _pixmap(svg_text, COLORS[token], size, scale, gap)
            if pixmap is None:
                return QtGui.QIcon()
            result.addPixmap(pixmap, mode, state)
    return result


def apply(button, name, size=SIZE_COMPACT, role="default"):
    """Put an icon on a button next to its text label and keep the icon size consistent."""
    button.setIcon(icon(name, size, role))
    button.setIconSize(QtCore.QSize(size + GAP, size))
