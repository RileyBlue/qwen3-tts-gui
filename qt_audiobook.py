"""Audiobook tab: a chapter script, a list of segments, and one editor for the selected segment(s); render one file.

Layout (Phase A1): a fixed header row (title, project actions, Render chapter) above a body with two columns.
Left: the chapter script, who reads it, and the render settings. Right: the segment list and the editor for what is
selected. The segments themselves are plain dicts (text, voice, kind, direction) exactly as before, so projects,
the cache and rendering are untouched.
"""

import os
import textwrap

from PySide6 import QtCore, QtGui, QtWidgets

import audiobook
import icons
from audio_utils import write_wav
from config import DEFAULT_LANGUAGE, LANGUAGES, OUTPUT_DIR
from theme import COLORS, FONT_PX, set_role, style_numeric
from voice_list import NAME_ROLE, VoiceList
from voice_lock import EMOTION_PRESETS

DESC_ROLE, ROLE_ROLE, NUM_ROLE = QtCore.Qt.UserRole + 2, QtCore.Qt.UserRole + 3, QtCore.Qt.UserRole + 4
ROW_H = 30          # one line per segment: about twice the rows of the two-line My Voices anatomy, same type size
NO_TEXT = "(empty segment)"


class SegmentRowDelegate(QtWidgets.QStyledItemDelegate):
    """One compact line per segment:  12  N  Ana   [sad] The rain fell on the old harbour town…

    number | N (narration) or D (dialogue) | short voice | text excerpt. Everything is painted from the item's data, so
    the rows stay ordinary list items: selection, keyboard handling and the accessibility tree come from Qt, and the
    item's accessible text reads the whole row. The same type size as the rest of the app; only the second line is gone."""

    NUM_W, ROLE_W, VOICE_MAX, PAD = 30, 22, 130, 12

    def sizeHint(self, option, index):
        return QtCore.QSize(1, ROW_H)

    def paint(self, painter, option, index):
        State = QtWidgets.QStyle
        selected = bool(option.state & State.State_Selected)
        hovered = bool(option.state & State.State_MouseOver)
        rect = option.rect
        painter.save()
        painter.fillRect(rect, QtGui.QColor(COLORS["selection"] if selected else COLORS["surface_hover"] if hovered
                                            else COLORS["surface"]))
        painter.fillRect(QtCore.QRect(rect.left(), rect.bottom(), rect.width(), 1), QtGui.QColor(COLORS["border"]))
        if selected:   # selection is shape as well as colour: an accent bar on the left edge
            painter.fillRect(QtCore.QRect(rect.left(), rect.top(), 3, rect.height() - 1), QtGui.QColor(COLORS["accent"]))
        font = QtGui.QFont(option.font)
        font.setPixelSize(FONT_PX["body"])
        bold = QtGui.QFont(font)
        bold.setWeight(QtGui.QFont.DemiBold)
        muted = QtGui.QColor(COLORS["text_secondary"] if selected else COLORS["text_muted"])
        text_color = QtGui.QColor(COLORS["text"])
        inner = rect.adjusted(self.PAD, 0, -self.PAD, -1)
        x = inner.left()

        def cell(width, align=QtCore.Qt.AlignLeft):
            return QtCore.QRect(x, rect.top(), width, rect.height() - 1), align | QtCore.Qt.AlignVCenter
        painter.setFont(font)
        painter.setPen(muted)
        box, align = cell(self.NUM_W - 6, QtCore.Qt.AlignRight)
        painter.drawText(box, align, str(index.data(NUM_ROLE) or ""))
        x += self.NUM_W
        role = index.data(ROLE_ROLE) or ""
        painter.setPen(text_color if role == "D" else muted)
        painter.setFont(bold)
        box, align = cell(self.ROLE_W, QtCore.Qt.AlignHCenter)
        painter.drawText(box, align, role)
        x += self.ROLE_W
        metrics = QtGui.QFontMetrics(bold)
        voice_w = min(self.VOICE_MAX, max(60, inner.width() // 4))
        painter.setPen(text_color)
        box, align = cell(voice_w)
        painter.drawText(box, align, metrics.elidedText(index.data(NAME_ROLE) or "", QtCore.Qt.ElideRight, voice_w))
        x += voice_w + 10
        painter.setFont(font)
        painter.setPen(muted if selected else QtGui.QColor(COLORS["text_secondary"]))
        room = max(20, inner.right() - x)
        box, align = cell(room)
        excerpt = QtGui.QFontMetrics(font).elidedText(index.data(DESC_ROLE) or "", QtCore.Qt.ElideRight, room)
        painter.drawText(box, align, excerpt)
        if option.state & State.State_HasFocus:
            painter.setPen(QtGui.QPen(QtGui.QColor(COLORS["accent"]), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rect.adjusted(1, 1, -2, -2))
        painter.restore()


class _Combo(QtWidgets.QComboBox):
    """Never wider than the page asks, ignores the mouse wheel (so scrolling the page can't silently change a voice),
    and shows the full text as a tooltip when the box elides it."""

    def __init__(self, parent=None, min_chars=8):
        super().__init__(parent)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(min_chars)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.currentIndexChanged.connect(lambda _i: not self.isEditable() and self.setToolTip(self.currentText()))

    def showPopup(self):
        view = self.view()    # the list may be wider than the (elided) box, so full voice names stay readable
        view.setMinimumWidth(min(view.sizeHintForColumn(0) + 40, 640))
        super().showPopup()

    def wheelEvent(self, event):
        event.ignore()    # scrolling the page must never change a voice, focused or not; the keyboard and the popup do


class _WheelGuard(QtCore.QObject):
    """Event filter: the wheel over a setting never changes it (a changed setting would re-render every segment)."""

    def eventFilter(self, obj, event):
        return event.type() == QtCore.QEvent.Wheel


def _field(text, buddy=None):
    label = QtWidgets.QLabel(text)
    set_role(label, "field")
    if buddy is not None:
        label.setBuddy(buddy)
    return label


class AudiobookTab(QtWidgets.QWidget):
    def __init__(self, win, text_edit, combo, seed_row, header=None):
        super().__init__()
        self.win = win
        self.segments = []
        self._loading = False
        self._labels = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 12)
        outer.setSpacing(12)

        # ---- header: what this page is, and the chapter-level actions (never scrolls away)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        if header is not None:
            top.addWidget(header("Audiobooks", "Split a chapter, choose voices, render one audio file."), 1)
        else:
            top.addWidget(QtWidgets.QLabel("Audiobooks"), 1)
        self.btn_save = QtWidgets.QPushButton("Save project…")
        self.btn_save.clicked.connect(self.save_project)
        self.btn_load = QtWidgets.QPushButton("Load project…")
        self.btn_load.clicked.connect(self.load_project)
        self.btn_render = QtWidgets.QPushButton("Render chapter")
        set_role(self.btn_render, "primary")
        self.btn_render.setToolTip("Unchanged segments are cached, so rendering again after an edit only regenerates what changed.")
        self.btn_render.clicked.connect(self.render_chapter)
        for b in (self.btn_save, self.btn_load, self.btn_render):
            top.addWidget(b, 0, QtCore.Qt.AlignVCenter)
        outer.addLayout(top)

        # ---- body (scrolls only if the window is smaller than the minimum)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        cols = QtWidgets.QHBoxLayout(body)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(24)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        cols.addLayout(self._build_script_column(text_edit, seed_row), 5)
        cols.addLayout(self._build_segment_column(text_edit), 6)

        self._chain()
        self.refresh_voices()
        self.rebuild()

    # ------------------------------------------------------------------ layout
    def _build_script_column(self, text_edit, seed_row):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        self.txt_source = text_edit(5, "Paste the chapter here, then choose Split into segments.")
        self.txt_source.setAccessibleName("Chapter script")
        self.txt_source.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        col.addWidget(_field("Chapter script", self.txt_source))
        col.addWidget(self.txt_source, 1)

        self.cmb_narrator = _Combo()
        self.cmb_dialogue = _Combo()
        self.cmb_lang = _Combo(min_chars=8)
        self.cmb_lang.addItems(LANGUAGES)
        self.cmb_lang.setCurrentText(DEFAULT_LANGUAGE)
        for w, name in ((self.cmb_narrator, "Narrator voice"), (self.cmb_dialogue, "Dialogue voice"),
                        (self.cmb_lang, "Language")):
            w.setAccessibleName(name)
        cast = QtWidgets.QGridLayout()
        cast.setHorizontalSpacing(12)
        cast.setVerticalSpacing(4)
        for col_index, (label, w) in enumerate((("Narrator voice", self.cmb_narrator), ("Dialogue voice", self.cmb_dialogue))):
            cast.addWidget(_field(label, w), 0, col_index)
            cast.addWidget(w, 1, col_index)
        cast.addWidget(_field("Language", self.cmb_lang), 2, 0)
        cast.addWidget(self.cmb_lang, 3, 0)
        self.btn_split = QtWidgets.QPushButton("Split into segments")
        self.btn_split.setToolTip("Quoted speech becomes dialogue segments; everything else is narration.")
        self.btn_split.clicked.connect(self.auto_split)
        cast.addWidget(self.btn_split, 3, 1)
        cast.setColumnStretch(0, 1)
        cast.setColumnStretch(1, 1)
        col.addLayout(cast)
        hint = QtWidgets.QLabel("Quoted speech becomes dialogue.")   # one line: no height-for-width, so the page never grows with the window width
        set_role(hint, "caption")
        col.addWidget(hint)

        # ---- render settings (collapsed: the defaults are right for most chapters)
        self.btn_settings = QtWidgets.QPushButton("Render settings")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setAccessibleName("Render settings")
        set_role(self.btn_settings, "quiet")
        icons.apply(self.btn_settings, "chevron-right")
        self.btn_settings.setAccessibleDescription("Collapsed")
        self.btn_settings.toggled.connect(self._toggle_settings)
        col.addWidget(self.btn_settings, 0, QtCore.Qt.AlignLeft)

        self.pnl_settings = QtWidgets.QWidget()
        form = QtWidgets.QGridLayout(self.pnl_settings)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.spn_pause = QtWidgets.QSpinBox()
        self.spn_pause.setRange(0, 5000)
        self.spn_pause.setSingleStep(50)
        self.spn_pause.setSuffix(" ms")
        self.spn_pause.setValue(350)
        style_numeric(self.spn_pause, width="fit", name="Pause between segments")
        self.chk_loudness = QtWidgets.QCheckBox("Match loudness between voices")
        self.chk_loudness.setChecked(True)
        self.cmb_seed_mode, self.spn_seed = seed_row("Fixed", 42)
        self.cmb_seed_mode.setAccessibleName("Seed mode")
        self.spn_seed.setAccessibleName("Seed value")
        self.spn_temp = QtWidgets.QDoubleSpinBox()
        self.spn_temp.setRange(0.1, 1.5)
        self.spn_temp.setSingleStep(0.05)
        self.spn_temp.setValue(0.7)
        style_numeric(self.spn_temp, width="fit", name="Clone temperature")
        form.addWidget(_field("Pause between segments", self.spn_pause), 0, 0)
        form.addWidget(self.spn_pause, 0, 1)
        form.addWidget(self.chk_loudness, 1, 0, 1, 2)
        form.addWidget(_field("Seed", self.cmb_seed_mode), 2, 0)
        seed = QtWidgets.QHBoxLayout()
        seed.addWidget(self.cmb_seed_mode)
        seed.addWidget(self.spn_seed)
        form.addLayout(seed, 2, 1)
        form.addWidget(_field("Clone temperature", self.spn_temp), 3, 0)
        form.addWidget(self.spn_temp, 3, 1)
        form.setColumnStretch(1, 1)
        self._wheel_guard = _WheelGuard(self)
        for w in (self.spn_pause, self.cmb_seed_mode, self.spn_seed, self.spn_temp):
            w.installEventFilter(self._wheel_guard)
            if hasattr(w, "lineEdit") and w.lineEdit() is not None:
                w.lineEdit().installEventFilter(self._wheel_guard)
        note = QtWidgets.QLabel("Unchanged segments are cached, so rendering again after an edit only regenerates what "
                                "changed. Direction accepts any free text, like [tag] in Clone Voice.")
        note.setWordWrap(True)
        set_role(note, "caption")
        form.addWidget(note, 4, 0, 1, 2)
        self.pnl_settings.setVisible(False)
        col.addWidget(self.pnl_settings)
        return col

    def _build_segment_column(self, text_edit):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        self.lbl_count = QtWidgets.QLabel("Segments")
        set_role(self.lbl_count, "field")
        col.addWidget(self.lbl_count)
        self.listw = VoiceList()
        set_role(self.listw, "voices")
        self.listw.setItemDelegate(SegmentRowDelegate(self.listw))
        self.listw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listw.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.listw.setMouseTracking(True)
        self.listw.setAccessibleName("Segments")
        self.listw.setMinimumHeight(ROW_H * 3)
        self.listw.itemSelectionChanged.connect(self.on_selection)
        col.addWidget(self.listw, 1)

        title_row = QtWidgets.QHBoxLayout()
        self.lbl_seg_title = QtWidgets.QLabel("Selected segment")
        set_role(self.lbl_seg_title, "field")
        title_row.addWidget(self.lbl_seg_title, 1)
        self.btn_preview = QtWidgets.QPushButton("▶ Preview segment")
        self.btn_preview.setAccessibleName("Preview segment")
        self.btn_preview.clicked.connect(self.preview_segment)
        title_row.addWidget(self.btn_preview)
        col.addLayout(title_row)
        self.txt_segment = text_edit(3)
        self.txt_segment.setMaximumHeight(self.txt_segment.fontMetrics().lineSpacing() * 3 + 16)   # the list gets the spare height
        self.txt_segment.setAccessibleName("Selected segment text")
        self.txt_segment.textChanged.connect(self.on_segment_text_changed)
        col.addWidget(self.txt_segment)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self.cmb_seg_voice = _Combo(min_chars=8)
        self.cmb_seg_voice.setAccessibleName("Voice for the selected segments")
        self.cmb_seg_voice.setPlaceholderText("Choose a voice")
        self.cmb_seg_voice.activated.connect(self.on_voice_chosen)
        self.cmb_seg_direction = _Combo(min_chars=6)
        self.cmb_seg_direction.setEditable(True)
        self.cmb_seg_direction.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.cmb_seg_direction.addItems([""] + list(EMOTION_PRESETS))
        self.cmb_seg_direction.setAccessibleName("Direction for the selected segment")
        self.cmb_seg_direction.lineEdit().setPlaceholderText("e.g. whisper, sad, or any style")
        # editTextChanged also covers auto-completion and picking a preset; programmatic refills run under _loading
        self.cmb_seg_direction.editTextChanged.connect(self.on_direction_edited)
        for label, w, stretch in (("Voice", self.cmb_seg_voice, 3), ("Direction", self.cmb_seg_direction, 2)):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(4)
            box.addWidget(_field(label, w))
            box.addWidget(w)
            row.addLayout(box, stretch)
        col.addLayout(row)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        specs = [
            ("btn_split_cursor", "Split here", "Split the segment at the text cursor", self.split_at_cursor),
            ("btn_merge", "Merge next", "Merge the segment with the next one", self.merge_next),
            ("btn_add", "Add after", "Add a new segment after the selection", self.add_after),
            ("btn_up", "Move up", "Move the segment up", lambda: self.move(-1)),
            ("btn_down", "Move down", "Move the segment down", lambda: self.move(1)),
            ("btn_delete", "Delete", "Delete the selected segments", self.delete_selected),
        ]
        for i, (attr, label, name, fn) in enumerate(specs):
            b = QtWidgets.QPushButton(label)
            b.setAccessibleName(name)
            b.setToolTip(name)
            b.clicked.connect(fn)
            setattr(self, attr, b)
            grid.addWidget(b, i // 3, i % 3)
        col.addLayout(grid)
        return col

    def _chain(self):
        order = [self.btn_save, self.btn_load, self.btn_render, self.txt_source, self.cmb_narrator, self.cmb_dialogue,
                 self.cmb_lang, self.btn_split, self.btn_settings, self.spn_pause, self.chk_loudness,
                 self.cmb_seed_mode, self.spn_seed, self.spn_temp, self.listw, self.btn_preview, self.txt_segment,
                 self.cmb_seg_voice, self.cmb_seg_direction, self.btn_split_cursor, self.btn_merge, self.btn_add,
                 self.btn_up, self.btn_down, self.btn_delete]
        for first, second in zip(order, order[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

    def _toggle_settings(self, on):
        self.pnl_settings.setVisible(on)
        icons.apply(self.btn_settings, "chevron-down" if on else "chevron-right")
        self.btn_settings.setAccessibleDescription("Expanded" if on else "Collapsed")

    # ------------------------------------------------------------------ voices
    def _fill_voices(self, cmb, current=None):
        """Refill a voice combo from the library; a voice that no longer exists stays visible as 'Missing: …'."""
        cmb.clear()
        for label, key in audiobook.voice_choices():
            cmb.addItem(label, key)
        if current:
            idx = cmb.findData(current)
            if idx < 0:
                cmb.addItem(f"Missing: {current}", current)
                idx = cmb.count() - 1
            cmb.setCurrentIndex(idx)
        else:
            cmb.setCurrentIndex(-1)

    def refresh_voices(self):
        self._loading = True
        try:
            choices = audiobook.voice_choices()
            self._labels = {key: label for label, key in choices}
            for cmb in (self.cmb_narrator, self.cmb_dialogue):
                current = cmb.currentData()
                cmb.clear()
                for label, key in choices:
                    cmb.addItem(label, key)
                idx = cmb.findData(current)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)
            if self.cmb_dialogue.currentData() == self.cmb_narrator.currentData():
                for index in range(self.cmb_dialogue.count()):
                    key = self.cmb_dialogue.itemData(index)
                    if key.startswith("voice:") and key != self.cmb_narrator.currentData():
                        self.cmb_dialogue.setCurrentIndex(index)
                        break
        finally:
            self._loading = False
        self._refresh_items()
        self._sync_voice_combo()      # not the whole editor: a refill must not reset the text caret or undo history

    def _voice_label(self, key):
        return self._labels.get(key) or (f"Missing: {key}" if key else "No voice")

    # ------------------------------------------------------------------ list <-> segments
    @staticmethod
    def _preview(text):
        flat = " ".join(text.split())
        return flat if len(flat) <= 140 else flat[:137] + "…"

    def _short_voice(self, key):
        """'Qwen: Ana Brightwater' -> 'Ana Brightwater'; other engines keep a short prefix ('XTTS Ana'). The full
        label is in the tooltip and the accessible text."""
        label = self._voice_label(key)
        if label.startswith("Missing"):
            return label
        return label[len("Qwen: "):] if label.startswith("Qwen: ") else label.replace(": ", " ", 1)

    def _style_item(self, item, i):
        seg = self.segments[i]
        label = self._voice_label(seg["voice"])
        direction = (seg.get("direction") or "").strip()
        preview = self._preview(seg["text"]) or NO_TEXT
        shown = f"[{direction}] {preview}" if direction else preview
        kind = seg["kind"].capitalize()
        item.setText(f"{i + 1} {label}")
        item.setData(NAME_ROLE, self._short_voice(seg["voice"]))
        item.setData(DESC_ROLE, shown)
        item.setData(NUM_ROLE, i + 1)
        item.setData(ROLE_ROLE, "D" if seg["kind"] == "dialogue" else "N")
        item.setData(QtCore.Qt.AccessibleTextRole,
                     f"Segment {i + 1}, {seg['kind']}, voice {label}" + (f", direction {direction}" if direction else "")
                     + f". {preview}")
        flat = " ".join(seg["text"].split())
        body = "\n".join(textwrap.wrap(flat[:400] + ("…" if len(flat) > 400 else ""), 70))
        item.setToolTip(f"Segment {i + 1} · {kind} · {label}" + (f" · {direction}" if direction else "") + f"\n{body}")

    def _refresh_items(self):
        self.listw.blockSignals(True)
        try:
            for i in range(self.listw.count()):
                self._style_item(self.listw.item(i), i)
        finally:
            self.listw.blockSignals(False)

    def rebuild(self, select=None):
        """Repopulate the list from self.segments (after split, merge, add, delete, move or load)."""
        self._loading = True
        self.listw.blockSignals(True)
        try:
            self.listw.clear()
            for i in range(len(self.segments)):
                item = QtWidgets.QListWidgetItem()
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                item.setSizeHint(QtCore.QSize(1, ROW_H))
                self.listw.addItem(item)
                self._style_item(item, i)
            n = len(self.segments)
            self.lbl_count.setText(f"Segments ({n})" if n else "Segments")
            if select is not None and n:
                self.listw.setCurrentRow(max(0, min(select, n - 1)))
        finally:
            self.listw.blockSignals(False)
            self._loading = False
        self.on_selection()

    def selected_rows(self):
        return sorted({self.listw.row(i) for i in self.listw.selectedItems()})

    def current_row(self):
        rows = self.selected_rows()
        return rows[0] if len(rows) == 1 else -1

    def _set_enabled(self, widget, on):
        if not on and widget.hasFocus():
            self.listw.setFocus()     # never leave focus on a control that just went away
        widget.setEnabled(on)

    def on_selection(self):
        self._fill_editor()

    def _fill_editor(self):
        rows = self.selected_rows()
        single = len(rows) == 1
        self._loading = True
        try:
            r = rows[0] if single else -1
            self.txt_segment.setPlainText(self.segments[r]["text"] if single else "")
            self.txt_segment.document().clearUndoRedoStacks()
            self.txt_segment.setPlaceholderText(
                "" if single else "Split the chapter script, or add a segment, to start." if not self.segments
                else "Select one segment to edit its text.")
            self._sync_voice_combo(rows)
            self.cmb_seg_direction.setCurrentText(self.segments[r].get("direction", "") if single else "")
            if not rows:
                self.lbl_seg_title.setText("Selected segment")
            elif single:
                self.lbl_seg_title.setText(f"Segment {r + 1} · {self.segments[r]['kind'].capitalize()}")
            else:
                self.lbl_seg_title.setText(f"{len(rows)} segments selected")
        finally:
            self._loading = False
        self.refresh_actions()

    def _sync_voice_combo(self, rows=None):
        was = self._loading
        self._loading = True
        try:
            rows = self.selected_rows() if rows is None else rows
            voices = {self.segments[i]["voice"] for i in rows}
            self._fill_voices(self.cmb_seg_voice, next(iter(voices)) if len(voices) == 1 else None)
        finally:
            self._loading = was

    def refresh_actions(self):
        """Enable only what the current selection (and a running task) allows; never touches the editor contents."""
        rows = self.selected_rows()
        single = len(rows) == 1
        self._set_enabled(self.txt_segment, single)
        self._set_enabled(self.cmb_seg_voice, bool(rows))
        self._set_enabled(self.cmb_seg_direction, single)
        self._set_enabled(self.btn_split_cursor, single)
        self._set_enabled(self.btn_merge, single and rows[0] < len(self.segments) - 1)
        self._set_enabled(self.btn_up, single and rows[0] > 0)
        self._set_enabled(self.btn_down, single and rows[0] < len(self.segments) - 1)
        self._set_enabled(self.btn_delete, bool(rows))
        self._set_enabled(self.btn_preview, single and not getattr(self.win, "busy", False))

    def on_voice_chosen(self, index):
        """The user picked a voice: it applies to every selected segment. Never fired by refills."""
        if self._loading:
            return
        key = self.cmb_seg_voice.itemData(index)
        rows = self.selected_rows()
        if not key or not rows:
            return
        for r in rows:
            self.segments[r]["voice"] = key
            self._style_item(self.listw.item(r), r)

    def on_direction_edited(self, text):
        if self._loading:
            return
        r = self.current_row()
        if r >= 0:
            self.segments[r]["direction"] = text.strip()
            self._style_item(self.listw.item(r), r)

    def on_segment_text_changed(self):
        if self._loading:
            return
        r = self.current_row()
        if r >= 0:
            self.segments[r]["text"] = self.txt_segment.toPlainText()
            self._style_item(self.listw.item(r), r)

    # ------------------------------------------------------------------ editing actions
    def auto_split(self):
        text = self.txt_source.toPlainText().strip()
        if not text:
            self.win.status("Error: paste some text first.")
            return
        if self.segments:
            answer = QtWidgets.QMessageBox.question(
                self, "Replace segments?", "Replace the current segments with a new split?")
            if answer != QtWidgets.QMessageBox.Yes:
                return
        narrator = self.cmb_narrator.currentData()
        dialogue = self.cmb_dialogue.currentData() or narrator
        self.segments = audiobook.split_narration_dialogue(text, narrator, dialogue)
        self.rebuild(select=0)
        self.win.status(f"Split into {len(self.segments)} segments. Assign voices, then render.")

    def split_at_cursor(self):
        r = self.current_row()
        if r < 0:
            return
        pos = self.txt_segment.textCursor().position()
        parts = audiobook.split_segment_at(self.segments[r], pos)
        if len(parts) == 2:
            self.segments[r:r + 1] = parts
            self.rebuild(select=r + 1)

    def merge_next(self):
        r = self.current_row()
        if 0 <= r < len(self.segments) - 1:
            self.segments[r:r + 2] = [audiobook.merge_segments(self.segments[r], self.segments[r + 1])]
            self.rebuild(select=r)

    def add_after(self):
        rows = self.selected_rows()
        r = rows[-1] if rows else len(self.segments) - 1
        voice = self.cmb_narrator.currentData() or ""
        self.segments.insert(r + 1, audiobook.new_segment("New segment.", voice))
        self.rebuild(select=r + 1)

    def delete_selected(self):
        rows = self.selected_rows()
        if not rows:
            return
        for r in reversed(rows):
            del self.segments[r]
        self.rebuild(select=rows[0])

    def move(self, delta):
        r = self.current_row()
        t = r + delta
        if r < 0 or not 0 <= t < len(self.segments):
            return
        self.segments[r], self.segments[t] = self.segments[t], self.segments[r]
        self.rebuild(select=t)

    # ------------------------------------------------------------------ rendering
    def render_settings(self):
        return dict(
            language=self.cmb_lang.currentText(),
            min_words=self.win.spn_min.value(), max_words=self.win.spn_max.value(),
            seed=self.spn_seed.value() if self.cmb_seed_mode.currentText() == "Fixed" else None,
            temperature=round(self.spn_temp.value(), 3),
        )

    def _check_segments(self, segs):
        if not segs:
            self.win.status("Error: there are no segments.")
            return False
        for s in segs:
            if not s["text"].strip():
                self.win.status("Error: a segment has no text.")
                return False
            if not s["voice"]:
                self.win.status("Error: every segment needs a voice.")
                return False
        if any(s["voice"].startswith("xtts:") for s in segs) and os.environ.get("COQUI_TOS_AGREED") != "1":
            answer = QtWidgets.QMessageBox.question(
                self, "XTTS model license",
                "XTTS-v2 is subject to the Coqui Public Model License. "
                "Review https://tts-hub.github.io/cpml before continuing.\n\n"
                "Do you have a commercial license or agree to the non-commercial license?")
            if answer != QtWidgets.QMessageBox.Yes:
                self.win.status("XTTS license acceptance is required.")
                return False
            os.environ["COQUI_TOS_AGREED"] = "1"
        if any(s["voice"].startswith("f5:") for s in segs) and os.environ.get("F5_TTS_LICENSE_CONFIRMED") != "1":
            answer = QtWidgets.QMessageBox.question(
                self, "F5-TTS model license",
                "The pretrained F5-TTS weights use the CC BY-NC 4.0 license. "
                "Review https://creativecommons.org/licenses/by-nc/4.0/ before continuing.\n\n"
                "Do you have permission to use these weights for your intended purpose?")
            if answer != QtWidgets.QMessageBox.Yes:
                self.win.status("F5-TTS license confirmation is required.")
                return False
            os.environ["F5_TTS_LICENSE_CONFIRMED"] = "1"
        return True

    def preview_segment(self):
        r = self.current_row()
        if r < 0:
            return
        seg = dict(self.segments[r])
        if not self._check_segments([seg]):
            return
        settings = self.render_settings()

        def work(task):
            return audiobook.render_segment(self.win.generator(task), seg, settings)

        def done(result):
            wav, sr = result
            if wav is None:
                self.win.status("Stopped before the segment was finished.")
                return
            path = os.path.join(OUTPUT_DIR, "_segment_preview.wav")
            write_wav(path, wav, sr)
            self.win.status(f"Previewing segment {r + 1}.")
            self.win.play(path)

        self.win.run_task(work, done)

    def render_chapter(self):
        segs = [dict(s) for s in self.segments]
        if not self._check_segments(segs):
            return
        settings = self.render_settings()
        pause, loudness = self.spn_pause.value(), self.chk_loudness.isChecked()

        def work(task):
            gen = self.win.generator(task)
            results = audiobook.render_chapter(
                gen, segs, settings, should_stop=task.stop_event.is_set, notify=task.notify.emit)
            if results is None:
                return None
            return audiobook.compile_audio(results, pause_ms=pause, match_loudness=loudness)

        def done(result):
            if result is None:
                self.win.status("Stopped. Finished segments are cached; render again to resume.")
                return
            wav, sr = result
            path = self.win.output_path("Audiobook")
            write_wav(path, wav, sr)
            self.win.last_output = path
            self.win.btn_play_last.setEnabled(True)
            self.win.status(f"Chapter rendered ({len(segs)} segments, {len(wav) / sr / 60:.1f} min): {path}")
            print(f"Audiobook generation and compilation complete: {path}", flush=True)
            QtWidgets.QApplication.beep()

        self.win.run_task(work, done)

    # ------------------------------------------------------------------ project files
    def project_meta(self):
        return dict(source=self.txt_source.toPlainText(), language=self.cmb_lang.currentText(),
                    pause_ms=self.spn_pause.value(), loudness=self.chk_loudness.isChecked())

    def save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", "", "Project (*.json)")
        if path:
            audiobook.save_project(path, self.segments, self.project_meta())
            self.win.status(f"Project saved: {path}")

    def load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load project", "", "Project (*.json)")
        if not path:
            return
        self.segments, meta = audiobook.load_project(path)
        self.txt_source.setPlainText(meta.get("source", ""))
        self.cmb_lang.setCurrentText(meta.get("language", DEFAULT_LANGUAGE))
        self.spn_pause.setValue(meta.get("pause_ms", 350))
        self.chk_loudness.setChecked(meta.get("loudness", True))
        self.rebuild(select=0)
        self.win.status(f"Project loaded: {path}")
