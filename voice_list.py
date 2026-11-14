"""My Voices list: a native QListWidget whose rows are painted as two lines.

Row anatomy:   Name .......................... Language · Created
               Description (one line, elided)

Why a delegate and not row widgets: the rows stay ordinary model items, so keyboard handling,
multi-selection, type-ahead search and the accessibility tree (rows and their names) come from Qt
itself. Each item carries an explicit accessible name that reads the whole row, and a tooltip with
the full description for the elided line. Nothing here reads or writes the voice files.
"""

from datetime import datetime

from PySide6 import QtCore, QtGui, QtWidgets

from theme import COLORS, FONT_PX

ROW_HEIGHT = 60
PAD_X = 16
SLUG_ROLE = QtCore.Qt.UserRole
NAME_ROLE = QtCore.Qt.UserRole + 1


def format_language(language):
    language = (language or "").strip()
    return language[:1].upper() + language[1:] if language else ""


def format_created(created):
    """'2026-11-10T09:00:00' -> 'Nov 10, 2026' in the user's locale; unparseable text is shown as is."""
    created = (created or "").strip()
    if not created:
        return ""
    try:
        stamp = datetime.fromisoformat(created)
    except ValueError:
        return created
    return QtCore.QLocale().toString(QtCore.QDate(stamp.year, stamp.month, stamp.day), "MMM d, yyyy")


def row_meta(detail):
    return " · ".join(part for part in (format_language(detail.get("language")),
                                        format_created(detail.get("created"))) if part)


def accessible_text(detail):
    """What a screen reader says for the row: name, description, language, date."""
    parts = [detail["name"]]
    if detail.get("description"):
        parts.append(detail["description"])
    meta = row_meta(detail)
    if meta:
        parts.append(meta)
    return ". ".join(parts)


def make_item(detail):
    item = QtWidgets.QListWidgetItem(detail["name"])      # DisplayRole: type-ahead matches the name
    item.setData(SLUG_ROLE, detail["slug"])
    item.setData(NAME_ROLE, detail["name"])
    item.setData(QtCore.Qt.UserRole + 2, detail.get("description", ""))
    item.setData(QtCore.Qt.UserRole + 3, row_meta(detail))
    item.setData(QtCore.Qt.AccessibleTextRole, accessible_text(detail))
    if detail.get("description"):
        item.setToolTip(detail["description"])
    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)   # rename goes through the Rename action
    item.setSizeHint(QtCore.QSize(1, ROW_HEIGHT))
    return item


class VoiceList(QtWidgets.QListWidget):
    """QListWidget where Return/Enter activate the current row on every platform.

    Qt's macOS behaviour reserves Return for starting an edit (and Command+Down/Command+O for
    activation), so without this Return would do nothing on a Mac.
    """

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not event.modifiers() & ~QtCore.Qt.KeypadModifier:
            item = self.currentItem()
            if item is not None:
                self.itemActivated.emit(item)
            event.accept()
            return
        super().keyPressEvent(event)


class VoiceRowDelegate(QtWidgets.QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QtCore.QSize(1, ROW_HEIGHT)   # the list gives every row the full viewport width

    def paint(self, painter, option, index):
        State = QtWidgets.QStyle
        selected = bool(option.state & State.State_Selected)
        hovered = bool(option.state & State.State_MouseOver)
        focused = bool(option.state & State.State_HasFocus)
        rect = option.rect
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)

        background = COLORS["selection"] if selected else COLORS["surface_hover"] if hovered else COLORS["surface"]
        painter.fillRect(rect, QtGui.QColor(background))
        painter.fillRect(QtCore.QRect(rect.left(), rect.bottom(), rect.width(), 1), QtGui.QColor(COLORS["border"]))
        if selected:   # selection is shape as well as colour: an accent bar on the left edge
            painter.fillRect(QtCore.QRect(rect.left(), rect.top(), 3, rect.height() - 1), QtGui.QColor(COLORS["accent"]))

        name_font = QtGui.QFont(option.font)
        name_font.setPixelSize(FONT_PX["body"])
        name_font.setWeight(QtGui.QFont.DemiBold)
        small_font = QtGui.QFont(option.font)
        small_font.setPixelSize(FONT_PX["secondary"])
        secondary = QtGui.QColor(COLORS["text_secondary"])
        meta_color = secondary if selected else QtGui.QColor(COLORS["text_muted"])

        inner = rect.adjusted(PAD_X, 0, -PAD_X, -1)
        line1 = QtCore.QRect(inner.left(), rect.top() + 9, inner.width(), 20)
        line2 = QtCore.QRect(inner.left(), rect.top() + 31, inner.width(), 18)

        meta = index.data(QtCore.Qt.UserRole + 3) or ""
        painter.setFont(small_font)
        small = QtGui.QFontMetrics(small_font)
        meta_width = small.horizontalAdvance(meta) if meta else 0
        if meta:
            painter.setPen(meta_color)
            painter.drawText(line1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, meta)

        painter.setFont(name_font)
        painter.setPen(QtGui.QColor(COLORS["text"]))
        name_metrics = QtGui.QFontMetrics(name_font)
        name_room = max(40, line1.width() - meta_width - (16 if meta else 0))
        name = name_metrics.elidedText(index.data(NAME_ROLE) or "", QtCore.Qt.ElideRight, name_room)
        painter.drawText(line1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, name)

        description = (index.data(QtCore.Qt.UserRole + 2) or "").replace("\n", " ")
        painter.setFont(small_font)
        painter.setPen(secondary)
        painter.drawText(line2, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                         small.elidedText(description or "No description", QtCore.Qt.ElideRight, line2.width()))

        if focused and option.state & State.State_HasFocus:
            painter.setPen(QtGui.QPen(QtGui.QColor(COLORS["accent"]), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rect.adjusted(1, 1, -2, -2))
        painter.restore()
