"""Qt (PySide6) front end for Qwen3-TTS.

All model work lives in tts_generator / voice_library / model_loader; this file
only builds widgets and runs the backend on worker threads.
"""

import json
import os
import random
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import audio_player
import icons
import task_errors
import voice_library
import voice_list
from audio_utils import save_outputs, write_wav, load_wav
from config import (
    DEFAULT_ATTN, DEFAULT_DEVICE, DEFAULT_DTYPE, DEFAULT_LANGUAGE, DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS, ATTNS, DTYPES, LANGUAGES, MODELS, MODEL_MAP, OUTPUT_DIR,
)
from model_loader import ModelManager
from qt_audiobook import AudiobookTab
from theme import COLORS, SIDEBAR_WIDTH, apply_theme, set_role
from tts_generator import TTSGenerator
from voice_lock import EMOTION_PRESETS

DIRECTION_HELP = (
    "Direct the performance like an actor: put a [tag] before a passage and it applies until the next "
    "tag. Use a preset ([angry], [whisper], [sad]…) or any free-text style, e.g. "
    "[shaky voice, holding back tears]. Emphasis is shaped by wording and punctuation (!, ?, …, "
    "short sentences); Qwen3-TTS has no markup for stressing single words."
)

# Optional starting points for the voice description. Selecting one appends its phrase;
# deselecting (or editing it away) removes only that phrase, so typed text is never lost.
SUGGESTIONS = (
    ("Narration", "Clear, steady narrator voice, ideal for narration."),
    ("Relaxing", "Soft, slow, soothing and relaxed delivery."),
    ("Epic", "Deep, powerful, dramatic trailer-style voice."),
    ("Professional", "Polished, confident, neutral professional tone."),
)

# Create Voice content stops growing past this width so wide windows leave whitespace, not stretched fields.
CONTENT_MAX_WIDTH = 1120
PAGE_CREATE = 0

# Starts a Create Voice generation. Qt maps Ctrl to Command on macOS, so show the native spelling.
# Lucide icon for each sidebar entry; the text label always stays next to it.
NAV_ICONS = {"Create Voice": "wand-sparkles", "Clone Voice": "copy", "My Voices": "mic-vocal",
             "Audiobooks": "book-open", "Settings": "settings"}
GENERATE_SHORTCUT = "Ctrl+Return"
READY_STATUS = "Sample ready. Listen, then name and save the voice if you like it."
STOP_AUDIO_TIP = "Stop the audio that is playing"
STALE_STATUS = "Showing the previous sample. Generate voice to hear your changes."
# Create Voice generation stages. Only stages the backend really reports are shown: the model
# manager announces "Loading <kind> model…" (a first load may download it, and it does not say which),
# and the app itself knows when the model call has returned and generation starts.
STAGE_PREPARING = "Preparing…"
STAGE_LOADING = "Loading the voice model…"
STAGE_GENERATING = "Generating the sample…"
LOADING_HINT = "The first generation may download the voice model. This is required once."
GENERATING_HINT = "This may take several minutes, depending on your device and sample length."
FOOTER_CREATING = "Creating the sample…"
SEEK_STEP = 5.0   # seconds for the back / forward buttons


def shortcut_text(sequence=GENERATE_SHORTCUT):
    return QtGui.QKeySequence(sequence).toString(QtGui.QKeySequence.NativeText)


# Preview panel pages (Create Voice).
PV_EMPTY, PV_BUSY, PV_READY, PV_ERROR = range(4)


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class Task(QtCore.QObject):
    """Run a function on a plain thread and report back on the GUI thread."""

    notify = QtCore.Signal(str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.failure = None   # task_errors.Failure of the last exception: type, message, traceback

    def start(self, fn):
        def run():
            try:
                self.done.emit(fn(self))
            except BaseException as e:  # a worker thread cannot propagate; never leave the UI busy
                self.failure = task_errors.Failure.from_exception(e)
                task_errors.log_failure(self.failure)   # full traceback to the console
                self.failed.emit(self.failure.message)
        threading.Thread(target=run, daemon=True).start()


def text_edit(height_lines, placeholder="", text="", read_only=False):
    """QPlainTextEdit gives native undo/redo (Ctrl+Z, Ctrl+Y/Ctrl+Shift+Z) and Ctrl+A."""
    w = QtWidgets.QPlainTextEdit()
    w.setUndoRedoEnabled(True)
    w.setTabChangesFocus(True)  # Tab moves on instead of inserting a tab character (keyboard trap)
    w.setPlaceholderText(placeholder)
    w.setReadOnly(read_only)
    if text:
        w.setPlainText(text)
        w.document().clearUndoRedoStacks()
    fm = w.fontMetrics()
    w.setMinimumHeight(fm.lineSpacing() * height_lines + 16)
    return w


def combo(items, current=None):
    c = QtWidgets.QComboBox()
    c.addItems(items)
    if current is not None:
        c.setCurrentText(current)
    return c


def seed_row(default_mode, default_value):
    mode = combo(["Fixed", "Random"], default_mode)
    value = QtWidgets.QSpinBox()
    value.setRange(0, 2**31 - 1)
    value.setValue(default_value)
    value.setEnabled(default_mode == "Fixed")
    mode.currentTextChanged.connect(lambda m: value.setEnabled(m == "Fixed"))
    return mode, value


def tab_page(body_layout_builder):
    """Return (page, body_layout, bar_layout): scrollable body + fixed action bar."""
    page = QtWidgets.QWidget()
    outer = QtWidgets.QVBoxLayout(page)
    outer.setContentsMargins(24, 24, 24, 16)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    body = QtWidgets.QWidget()
    body_layout = QtWidgets.QVBoxLayout(body)
    scroll.setWidget(body)
    outer.addWidget(scroll, 1)
    bar = QtWidgets.QHBoxLayout()
    outer.addLayout(bar)
    return page, body_layout, bar


def page_header(title, subtitle=None):
    """Screen heading: one title and at most one short description."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 8)
    lay.setSpacing(4)
    lbl = QtWidgets.QLabel(title)
    set_role(lbl, "title")
    lay.addWidget(lbl)
    if subtitle:
        sub = QtWidgets.QLabel(subtitle)
        sub.setWordWrap(True)
        set_role(sub, "subtitle")
        lay.addWidget(sub)
    return box


def format_duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class WrapLabel(QtWidgets.QLabel):
    """Word-wrapped label that reserves the height its text needs at its current width.

    Box layouts don't always hand a width to height-for-width children, which clips wrapped text.
    """

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)

    def setText(self, text):
        super().setText(text)
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        height = self.heightForWidth(self.width()) if self.width() > 0 else 0
        if height > 0 and height != self.minimumHeight():
            self.setMinimumHeight(height)


class PageSwitcher(QtWidgets.QWidget):
    """Shows one page at a time. Unlike QStackedWidget, hidden pages take no part in layout, so the
    container is exactly as tall as the visible page (a stack sizes itself to its tallest page)."""

    def __init__(self):
        super().__init__()
        self._pages = []
        self._index = -1
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    def insertWidget(self, index, widget):
        self._pages.insert(index, widget)
        self.layout().addWidget(widget)
        widget.setVisible(False)

    def setCurrentIndex(self, index):
        self._index = index
        for i, page in enumerate(self._pages):
            page.setVisible(i == index)

    def currentIndex(self):
        return self._index

    def currentWidget(self):
        return self._pages[self._index]

    def widget(self, index):
        return self._pages[index]

    def count(self):
        return len(self._pages)


class WaveformView(QtWidgets.QWidget):
    """Peak bars drawn from the real generated samples. The played part takes the accent colour and a
    playhead marks the real playback position; a click seeks (the keyboard uses the seek slider)."""

    BARS = 48
    seek_requested = QtCore.Signal(float)   # 0..1 of the sample's length

    def __init__(self):
        super().__init__()
        self._peaks = []
        self._progress = None   # None: nothing is playing or paused, so no playhead
        self.setFixedHeight(56)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAccessibleName("Waveform of the generated sample")

    def set_samples(self, wav):
        a = np.abs(np.asarray(wav, dtype=np.float32)).reshape(-1)
        peaks = [float(c.max()) for c in np.array_split(a, self.BARS) if c.size] if a.size else []
        top = max(peaks, default=0) or 1.0
        self._peaks = [p / top for p in peaks]
        self.update()

    def set_progress(self, fraction):
        fraction = None if fraction is None else min(1.0, max(0.0, fraction))
        if fraction != self._progress:
            self._progress = fraction
            self.update()

    def mousePressEvent(self, event):
        if self._peaks and event.button() == QtCore.Qt.LeftButton and self.width() > 0:
            self.seek_requested.emit(min(1.0, max(0.0, event.position().x() / self.width())))

    def paintEvent(self, _event):
        if not self._peaks:
            return
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.NoPen)
        played, rest = QtGui.QColor(COLORS["accent"]), QtGui.QColor(COLORS["text_muted"])
        step = self.width() / len(self._peaks)
        bar = max(2.0, step - 3)
        mid = self.height() / 2
        head = None if self._progress is None else self._progress * self.width()
        for i, p in enumerate(self._peaks):
            h = max(3.0, p * self.height())
            painter.setBrush(played if head is not None and i * step + bar / 2 <= head else rest)
            painter.drawRoundedRect(QtCore.QRectF(i * step, mid - h / 2, bar, h), 1.5, 1.5)
        if head is not None:
            painter.setBrush(QtGui.QColor(COLORS["text"]))
            painter.drawRect(QtCore.QRectF(min(head, self.width() - 2), 0, 2, self.height()))


class ActivityBar(QtWidgets.QWidget):
    """Restrained indeterminate indicator: one short segment gliding along a quiet track.

    It says only "work is happening". It has no value, so it can never suggest a percentage or an
    estimate (a stock indeterminate QProgressBar wraps around and reads as two segments). The animation
    runs only while the widget is visible.
    """

    indeterminate = True

    def __init__(self):
        super().__init__()
        self.setFixedHeight(4)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAccessibleName("Activity in progress")
        self._t = 0.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1800)
        self._anim.setLoopCount(-1)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._anim.valueChanged.connect(self._step)

    def _step(self, value):
        self._t = float(value)
        self.update()

    def showEvent(self, event):
        self._anim.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._anim.stop()
        super().hideEvent(event)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        radius = self.height() / 2
        painter.setBrush(QtGui.QColor(COLORS["border"]))
        painter.drawRoundedRect(self.rect(), radius, radius)
        width = self.width() * 0.28
        phase = 1 - abs(2 * self._t - 1)          # glide out and back, never wrapping around
        painter.setBrush(QtGui.QColor(COLORS["accent"]))
        painter.drawRoundedRect(QtCore.QRectF((self.width() - width) * phase, 0, width, self.height()), radius, radius)


class DirectionBar(QtWidgets.QWidget):
    """Insert an [emotion] tag at the cursor of a text editor."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel("Direction:"))
        self.preset = combo(list(EMOTION_PRESETS))
        lay.addWidget(self.preset)
        btn = QtWidgets.QPushButton("Insert [tag] at cursor")
        btn.clicked.connect(self.insert)
        lay.addWidget(btn)
        lay.addStretch(1)

    def insert(self):
        self.editor.insertPlainText(f"[{self.preset.currentText()}] ")
        self.editor.setFocus()


class MainWindow(QtWidgets.QMainWindow):
    _failure_details = ""   # sanitized technical text of the last Create Voice failure
    _gen_t0 = None          # time.monotonic() when the running generation started
    _gen_stage = STAGE_PREPARING

    def __init__(self, models=None):
        super().__init__()
        self.setWindowTitle("Qwen3-TTS")
        self.models = models or ModelManager(self.model_settings)
        self.task = None
        self.busy = False
        self.pending_voice = None
        self.last_output = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # ---- menu bar
        menu = self.menuBar()
        tools = menu.addMenu("Tools")
        act_voice_mgr = tools.addAction("Voice Manager…")
        act_voice_mgr.triggered.connect(self.open_voice_manager)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        shell = QtWidgets.QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        # Widgets other pages depend on (status label, settings spin boxes) are built first.
        settings_page = self._build_settings_page()
        footer = self._build_footer()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_design_tab())
        clone_page = self._build_clone_tab()
        voices_page = self._build_voices_tab()
        self.audiobook_tab = AudiobookTab(self, text_edit, combo, seed_row)
        for dense in (clone_page, self.audiobook_tab):  # not redesigned yet: keep them compact
            dense.setProperty("density", "compact")
        self.stack.addWidget(clone_page)
        self.stack.addWidget(voices_page)
        lay = self.audiobook_tab.layout()
        lay.setContentsMargins(24, 24, 24, 16)
        lay.insertWidget(0, page_header("Audiobooks"))
        ab_scroll = QtWidgets.QScrollArea()
        ab_scroll.setWidgetResizable(True)
        ab_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        ab_scroll.setWidget(self.audiobook_tab)
        self.stack.addWidget(ab_scroll)
        self.stack.addWidget(settings_page)

        shell.addWidget(self._build_sidebar(("Create Voice", "Clone Voice", "My Voices", "Audiobooks"), "Settings"))
        main_col = QtWidgets.QVBoxLayout()
        main_col.setContentsMargins(0, 0, 0, 0)
        main_col.setSpacing(0)
        main_col.addWidget(self.stack, 1)
        main_col.addWidget(footer)
        shell.addLayout(main_col, 1)
        self.show_page(0)
        self.refresh_voice_lists()

        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(1024, screen.width()), min(640, screen.height()))
        self.resize(min(1240, screen.width() - 40), min(860, screen.height() - 60))

        # Buttons take keyboard focus only, so a mouse click never leaves a focus ring on
        # the selected navigation item; the description field gets the initial focus.
        for b in self.findChildren(QtWidgets.QPushButton):
            b.setFocusPolicy(QtCore.Qt.TabFocus)
        self._sync_nav_focus()  # the loop above must not give every nav item a tab stop
        for area in self.findChildren(QtWidgets.QScrollArea):  # a scroll area is not a control: no invisible stop
            area.setFocusPolicy(QtCore.Qt.NoFocus)
        self._apply_tab_order()
        self.txt_vd_instruct.setFocus()

        # Ctrl+Return starts a Create Voice generation (a disabled button ignores the click).
        QtGui.QShortcut(QtGui.QKeySequence(GENERATE_SHORTCUT), self, activated=self._shortcut_generate)

    # ------------------------------------------------------------------ settings / status
    def model_settings(self):
        return (self.cmb_device.currentText(), self.cmb_dtype.currentText(), self.cmb_attn.currentText())

    def _build_sidebar(self, entries, settings_label):
        """Left navigation. Selection is a neutral surface; the accent is reserved for actions."""
        side = QtWidgets.QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(SIDEBAR_WIDTH)
        lay = QtWidgets.QVBoxLayout(side)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(4)
        brand = QtWidgets.QLabel("Qwen3-TTS Studio")
        set_role(brand, "brand")
        lay.addWidget(brand)
        lay.addSpacing(16)
        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []

        def add_nav(label):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            icons.apply(btn, NAV_ICONS.get(label, ""), icons.SIZE_NAV, "nav")
            btn.setToolTip(f"{label} (Ctrl+{len(self.nav_buttons) + 1})")
            set_role(btn, "nav")
            self.nav_group.addButton(btn, len(self.nav_buttons))
            self.nav_buttons.append(btn)
            lay.addWidget(btn)

        for label in entries:
            add_nav(label)
        lay.addStretch(1)
        add_nav(settings_label)
        self.nav_group.idClicked.connect(self.show_page)
        for i in range(len(self.nav_buttons)):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda i=i: self.show_page(i))
        return side

    def show_page(self, index):
        # Hiding the focused control would hand focus to the first control of the next page, which then
        # shows a focus ring nobody asked for (looks like a highlighted button after a mouse click).
        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None and self.stack.isAncestorOf(focused) and index != self.stack.currentIndex():
            focused.clearFocus()
        self.stack.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self._sync_nav_focus()
        # Keyboard focus on a navigation item that is no longer selected (it is no longer a tab stop)
        # would keep drawing its focus ring, e.g. Settings while Create Voice is open. Follow the selection.
        if focused in self.nav_buttons and focused is not self.nav_buttons[index]:
            self.nav_buttons[index].setFocus(QtCore.Qt.OtherFocusReason)
        self._update_footer()

    def _sync_nav_focus(self):
        # The navigation is one tab stop (the selected item); arrow keys move between items.
        for button in self.nav_buttons:
            button.setFocusPolicy(QtCore.Qt.TabFocus if button.isChecked() else QtCore.Qt.NoFocus)

    def _apply_tab_order(self):
        """Tab follows the visual task order: navigation, creation form, primary action, preview
        actions, secondary settings, footer. Qt skips hidden and disabled widgets, and only the
        checked navigation item is a stop (arrow keys move between items)."""
        chain = [
            *self.nav_buttons,
            self.txt_vd_instruct, *(chip for chip, _name, _phrase in self.vd_chips),
            self.txt_vd_sample, self.btn_vd_reset_text, self.btn_vd_generate,
            self.btn_vd_notice_details, self.txt_vd_spoken, self.btn_vd_edit_text,
            self.sld_vd_seek, self.btn_vd_back, self.btn_vd_play, self.btn_vd_forward, self.edt_vd_name, self.btn_vd_save, self.btn_vd_download,
            self.btn_vd_regenerate,
            self.btn_vd_retry, self.btn_vd_error_settings, self.btn_vd_details, self.btn_vd_copy_details,
            self.txt_vd_details,
            self.cmb_vd_lang, self.btn_vd_advanced,
            self.cmb_vd_seed_mode, self.spn_vd_seed, self.chk_vd_autosave, self.cmb_vd_name_gender,
            self.btn_vd_rand_name, self.btn_vd_open_settings,
            self.btn_play_last, self.btn_stop_audio, self.btn_open_out, self.btn_stop,
        ]
        for first, second in zip(chain, chain[1:]):
            QtWidgets.QWidget.setTabOrder(first, second)

    def _shortcut_generate(self):
        if self.stack.currentIndex() == 0:
            self.btn_vd_generate.click()

    def model_settings(self):
        return (self.cmb_device.currentText(), self.cmb_dtype.currentText(), self.cmb_attn.currentText())

    def _build_settings_page(self):
        """Application-wide options (same widgets and defaults as before), grouped by scope.

        Task options (language, seed, auto-save, naming) stay on the screen that uses them.
        """
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(32, 32, 32, 24)
        outer.setSpacing(16)
        outer.addWidget(page_header(
            "Settings", "Application-wide options. Language, seed and naming live on each screen."))
        self.cmb_device = combo(["cuda", "cpu"], DEFAULT_DEVICE if cuda_available() else "cpu")
        self.cmb_dtype = combo(DTYPES, DEFAULT_DTYPE)
        self.cmb_attn = combo(ATTNS, DEFAULT_ATTN)
        self.spn_min = QtWidgets.QSpinBox()
        self.spn_min.setRange(0, 5000)
        self.spn_min.setValue(int(DEFAULT_MIN_WORDS))
        self.spn_max = QtWidgets.QSpinBox()
        self.spn_max.setRange(0, 5000)
        self.spn_max.setValue(int(DEFAULT_MAX_WORDS))

        def section(title, description, rows):
            head = QtWidgets.QLabel(title)
            set_role(head, "heading")
            outer.addWidget(head)
            desc = QtWidgets.QLabel(description)
            desc.setWordWrap(True)
            desc.setMaximumWidth(560)
            set_role(desc, "muted")
            outer.addWidget(desc)
            form = QtWidgets.QFormLayout()
            form.setHorizontalSpacing(16)
            form.setVerticalSpacing(12)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            for label, w in rows:
                lbl = QtWidgets.QLabel(label)
                lbl.setBuddy(w)
                form.addRow(lbl, w)
            outer.addLayout(form)

        section("Compute", "Used by every screen. Models reload when these change.",
                (("Device", self.cmb_device), ("Dtype", self.cmb_dtype), ("Attention", self.cmb_attn)))
        outer.addSpacing(8)
        section("Long-form text", "Used by Clone Voice and Audiobooks. Voice creation is not split into chunks.",
                (("Min words per chunk", self.spn_min), ("Max words per chunk", self.spn_max)))
        outer.addSpacing(8)
        note = QtWidgets.QLabel(
            "Defaults are unchanged from earlier versions. flash_attention_2 targets NVIDIA GPUs with the "
            "flash-attn package installed; behavior on other hardware has not been verified here. "
            "Models are downloaded on first use.")
        note.setWordWrap(True)
        note.setMaximumWidth(560)
        set_role(note, "muted")
        outer.addWidget(note)
        outer.addStretch(1)
        return page

    def _build_footer(self):
        """Global status strip: progress, last message, and audio/output shortcuts."""
        footer = QtWidgets.QFrame()
        footer.setObjectName("footer")
        lay = QtWidgets.QVBoxLayout(footer)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(3)
        lay.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(24, 8, 24, 8)
        row.setSpacing(8)
        # Decorative glyph for the two states that benefit from one; the status text always says it too.
        self.lbl_status_icon = QtWidgets.QLabel()
        self.lbl_status_icon.setFixedSize(icons.SIZE_COMPACT, icons.SIZE_COMPACT)
        self.lbl_status_icon.setVisible(False)
        row.addWidget(self.lbl_status_icon)
        row.setSpacing(8)
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAccessibleName("Status")
        row.addWidget(self.lbl_status, 1)
        self.btn_play_last = QtWidgets.QPushButton("Play last output")
        self.btn_play_last.setEnabled(False)
        self.btn_play_last.clicked.connect(lambda: self.play(self.last_output))
        self.btn_stop_audio = QtWidgets.QPushButton("Stop audio")
        self.btn_stop_audio.clicked.connect(self.stop_audio)
        self.btn_open_out = QtWidgets.QPushButton("Open output folder")
        self.btn_open_out.clicked.connect(self.open_output_folder)
        self.btn_stop = QtWidgets.QPushButton("Stop generation")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_play_last.setToolTip("Play the most recent Clone Voice or Audiobook render")
        self.btn_stop_audio.setToolTip(STOP_AUDIO_TIP)
        self.btn_open_out.setToolTip("Open the folder where renders are saved")
        self.btn_stop.setToolTip("Stop the running render after its current chunk or segment")
        self.btn_stop.setVisible(False)
        for b, glyph in ((self.btn_play_last, "play"), (self.btn_stop_audio, "square"),
                         (self.btn_open_out, "folder-open"), (self.btn_stop, "square")):
            set_role(b, "quiet")
            icons.apply(b, glyph)
            row.addWidget(b)
        lay.addLayout(row)
        # Stop audio is only live while the player reports audio; the player is asked, never assumed.
        # Hidden (so neither visible nor focusable) unless the player has a session, but it keeps its
        # space so the footer doesn't shift when it appears.
        policy = self.btn_stop_audio.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.btn_stop_audio.setSizePolicy(policy)
        self.btn_stop_audio.setVisible(False)
        self.audio_timer = QtCore.QTimer(self)
        self.audio_timer.setInterval(250)
        self.audio_timer.timeout.connect(self._refresh_audio_state)
        self.audio_timer.start()
        # The Preview player follows audio_player's real state (position, pause, end) while it is on screen.
        self.player_timer = QtCore.QTimer(self)
        self.player_timer.setInterval(100)
        self.player_timer.timeout.connect(self._tick_player)
        self.player_timer.start()
        self.gen_timer = QtCore.QTimer(self)   # ticks only while a sample is being generated
        self.gen_timer.setInterval(1000)
        self.gen_timer.timeout.connect(self._tick_elapsed)
        return footer

    def status(self, msg):
        self.lbl_status.setText(msg)
        kind = "error" if msg.startswith(("Error", "Playback error", "Couldn't")) else (
            "ok" if msg == READY_STATUS else None)
        glyph = {"error": "circle-exclamation-point", "ok": "circle-check"}.get(kind)
        if glyph:
            self.lbl_status_icon.setPixmap(icons.icon(glyph, icons.SIZE_COMPACT, "error" if kind == "error" else "default", gap=0)
                                           .pixmap(icons.SIZE_COMPACT, icons.SIZE_COMPACT))
        self.lbl_status_icon.setVisible(bool(glyph))

    def _refresh_audio_state(self):
        session = audio_player.state() != "stopped"   # playing or paused, from any page
        if session == self.btn_stop_audio.isHidden():
            self.btn_stop_audio.setVisible(session)

    def stop_audio(self):
        audio_player.stop()
        self._pv_start = 0.0
        self._refresh_audio_state()
        self._refresh_player()

    def open_output_folder(self):
        """Open the renders folder; recreate it if it was removed, and say so if the system can't open it."""
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        except OSError as exc:
            self.status(f"Couldn't open the output folder: {exc}")
            return
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(OUTPUT_DIR)):
            self.status(f"Couldn't open the output folder: {OUTPUT_DIR}")

    def play(self, path, start=0.0):
        """Start `path`; returns the error text if it can't play, else None."""
        try:
            audio_player.play(path, start=start) if start > 0 else audio_player.play(path)
            self._refresh_audio_state()
            self._refresh_player()
        except Exception as e:
            self.status(f"Playback error: {e}")
            self._refresh_player()
            return str(e)
        return None

    def open_voice_manager(self):
        dlg = VoiceManagerDialog(self)
        dlg.exec()

    def request_stop(self):
        if self.task:
            self.task.stop_event.set()
            self.status("Stop requested…")

    def set_busy(self, busy, stoppable=True, footer_progress=True):
        self.busy = busy
        self.progress.setRange(0, 0 if busy and footer_progress else 1)
        # Only the clone loop and audiobook render check the stop flag; other tasks can't be cancelled.
        self.btn_stop.setEnabled(busy and stoppable)
        self.btn_stop.setVisible(busy and stoppable)
        for b in self.action_buttons:
            b.setEnabled(not busy)
        self._refresh_save_button()  # re-enabling everything must not enable Save without a sample or name

    def _update_footer(self):
        # Create Voice has its own Play; "last output" only applies to Clone Voice and Audiobook renders.
        self.btn_play_last.setVisible(self.stack.currentIndex() != 0)

    def run_task(self, fn, on_done, on_fail=None, on_notify=None, stoppable=True,
                 footer_progress=True, footer_status=True):
        """footer_progress / footer_status False: the screen shows this task's progress itself
        (Create Voice), so the footer neither animates nor repeats the stage text."""
        if self.busy:
            self.status("Busy: a task is already running.")
            return
        task = Task()
        self.task = task
        if footer_status:
            task.notify.connect(self.status)
        if on_notify:
            task.notify.connect(on_notify)

        def failed(message):
            self.set_busy(False)
            if footer_status:
                self.status(f"Error: {message}")
            if on_fail:
                on_fail(message)

        task.done.connect(lambda r: (self.set_busy(False), on_done(r)))
        task.failed.connect(failed)
        self.set_busy(True, stoppable, footer_progress)
        task.start(fn)

    def generator(self, task):
        return TTSGenerator(self.models, notify=task.notify.emit, should_stop=task.stop_event.is_set)

    def output_path(self, prefix):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(OUTPUT_DIR, f"{prefix}_{stamp}_{uuid.uuid4().hex[:3]}.wav")

    def finish_long_form(self, prefix):
        def on_done(result):
            wavs, sr = result
            if not wavs:
                self.status("Stopped before any audio was produced.")
                return
            path = self.output_path(prefix)
            save_outputs(wavs, sr, path)
            self.last_output = path
            self.btn_play_last.setEnabled(True)
            self.status(("Stopped early; partial audio saved to: " if self.task.stop_event.is_set()
                         else "Done. Audio saved to: ") + path)
        return on_done

    # ------------------------------------------------------------------ Voice Creator
    def _build_design_tab(self):
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        page.setWidget(body)
        frame = QtWidgets.QHBoxLayout(body)
        frame.setContentsMargins(32, 32, 32, 24)
        self.vd_content = QtWidgets.QWidget()
        self.vd_content.setMaximumWidth(CONTENT_MAX_WIDTH)
        frame.addWidget(self.vd_content, 100, QtCore.Qt.AlignTop)
        frame.addStretch(1)
        outer = QtWidgets.QVBoxLayout(self.vd_content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(24)

        outer.addWidget(page_header(
            "Create a voice with AI", "Describe how you want it to sound, then listen to a sample."))
        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(32)
        left = self._build_creation_column()
        # Model, Language and Advanced settings configure generation, so they sit under Generate voice
        # in the form column rather than spanning under the Preview.
        left.addSpacing(8)
        divider = QtWidgets.QFrame()
        set_role(divider, "divider")
        left.addWidget(divider)
        left.addLayout(self._build_options_strip())
        self.vd_advanced = self._build_advanced_panel()
        self.vd_advanced.setVisible(False)
        left.addWidget(self.vd_advanced)
        left.addStretch(1)
        cols.addLayout(left, 5)
        cols.addWidget(self._build_preview_panel(), 4, QtCore.Qt.AlignTop)
        outer.addLayout(cols)

        self.txt_vd_instruct.textChanged.connect(self._on_description_changed)
        self.cmb_vd_lang.currentTextChanged.connect(lambda _t: self._refresh_stale_note())
        for cmb in (self.cmb_device, self.cmb_dtype, self.cmb_attn):
            cmb.currentTextChanged.connect(lambda _t: self._update_compute_summary())
        self._update_compute_summary()
        self._set_preview(PV_EMPTY)
        return page

    # ---- left column: describe and generate
    def _build_creation_column(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(8)
        lbl = QtWidgets.QLabel("Voice description")
        self.lbl_vd_description = lbl
        set_role(lbl, "field")
        col.addWidget(lbl)
        self.txt_vd_instruct = text_edit(
            4, "e.g. A warm, calm voice with a friendly tone",
            "Warm, gentle female voice, calm and soothing tone.")
        self.txt_vd_instruct.setFixedHeight(112)
        self.txt_vd_instruct.setTabChangesFocus(True)  # Tab/Shift+Tab leave the field; Enter still adds a line
        self.txt_vd_instruct.setAccessibleName("Voice description")
        lbl.setBuddy(self.txt_vd_instruct)
        col.addWidget(self.txt_vd_instruct)
        self.lbl_vd_field_error = QtWidgets.QLabel("Describe how you want the voice to sound.")
        set_role(self.lbl_vd_field_error, "error-text")
        self.lbl_vd_field_error.setVisible(False)
        col.addWidget(self.lbl_vd_field_error)

        col.addSpacing(8)
        caption = QtWidgets.QLabel("Suggestions")
        set_role(caption, "caption")
        col.addWidget(caption)
        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(8)
        self.vd_chips = []
        for name, phrase in SUGGESTIONS:
            chip = QtWidgets.QPushButton(name)
            chip.setCheckable(True)
            chip.setToolTip(f"Add to the description: {phrase}")
            set_role(chip, "chip")
            chip.toggled.connect(lambda on, ph=phrase: self._toggle_suggestion(ph, on))
            self.vd_chips.append((chip, name, phrase))
            chips.addWidget(chip)
        chips.addStretch(1)
        col.addLayout(chips)

        col.addSpacing(8)
        sample_head = QtWidgets.QHBoxLayout()
        sample_lbl = QtWidgets.QLabel("Sample text")
        set_role(sample_lbl, "field")
        sample_head.addWidget(sample_lbl)
        sample_head.addStretch(1)
        self.btn_vd_reset_text = QtWidgets.QPushButton("Use standard text")
        set_role(self.btn_vd_reset_text, "quiet")
        self.btn_vd_reset_text.setToolTip("Restore the standard sample passage")
        self.btn_vd_reset_text.clicked.connect(lambda: self.txt_vd_sample.setPlainText(voice_library.SAMPLE_TEXT))
        sample_head.addWidget(self.btn_vd_reset_text)
        col.addLayout(sample_head)
        self.txt_vd_sample = text_edit(3, text=voice_library.SAMPLE_TEXT)
        self.txt_vd_sample.setFixedHeight(self.txt_vd_sample.fontMetrics().lineSpacing() * 3 + 20)
        self.txt_vd_sample.setAccessibleName("Sample text")
        self.txt_vd_sample.setAccessibleDescription("This is what the generated voice will say")
        self.txt_vd_sample.setToolTip("Longer text takes longer to generate.")
        sample_lbl.setBuddy(self.txt_vd_sample)
        self.txt_vd_sample.textChanged.connect(self._on_sample_text_changed)
        col.addWidget(self.txt_vd_sample)
        sample_hint = QtWidgets.QLabel("This is what the generated voice will say.")
        set_role(sample_hint, "muted")
        col.addWidget(sample_hint)

        col.addSpacing(16)
        gen = QtWidgets.QHBoxLayout()
        self.btn_vd_generate = QtWidgets.QPushButton("Generate voice")
        set_role(self.btn_vd_generate, "primary")
        icons.apply(self.btn_vd_generate, "audio-lines", icons.SIZE_COMPACT, "primary")
        self.btn_vd_generate.setMinimumWidth(180)
        self.btn_vd_generate.setToolTip(f"Generate voice ({shortcut_text()})")
        self.btn_vd_generate.setAccessibleDescription(f"Keyboard shortcut {shortcut_text()}")
        self.btn_vd_generate.clicked.connect(self.generate_sample)
        gen.addWidget(self.btn_vd_generate)
        gen.addSpacing(12)
        self.lbl_vd_shortcut = QtWidgets.QLabel(f"or press {shortcut_text()}")
        set_role(self.lbl_vd_shortcut, "muted")
        gen.addWidget(self.lbl_vd_shortcut)
        gen.addStretch(1)
        col.addLayout(gen)
        col.addStretch(1)
        return col

    # ---- right column: preview states
    def _build_preview_panel(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("previewPanel")
        self.preview_panel = panel
        panel.setMinimumWidth(340)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(20, 16, 20, 20)
        lay.setSpacing(12)

        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Preview")
        self.lbl_vd_preview_title = title
        set_role(title, "heading")
        head.addWidget(title)
        head.addStretch(1)
        lay.addLayout(head)

        self.vd_preview = PageSwitcher()
        lay.addWidget(self.vd_preview)

        # empty
        empty = QtWidgets.QWidget()
        em = QtWidgets.QVBoxLayout(empty)
        em.setContentsMargins(0, 0, 0, 0)
        em.setSpacing(4)
        msg = WrapLabel("Your generated sample will appear here.")
        set_role(msg, "notice")
        hint = WrapLabel("Describe a voice, then choose Generate voice.")
        set_role(hint, "muted")
        em.addWidget(msg)
        em.addWidget(hint)
        em.addStretch(1)
        self.vd_preview.insertWidget(PV_EMPTY, empty)

        # busy
        busy = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(busy)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        self.lbl_vd_busy = WrapLabel(STAGE_PREPARING)
        self.lbl_vd_busy.setAccessibleName("Generation status")
        bl.addWidget(self.lbl_vd_busy)
        self.vd_busy_bar = ActivityBar()
        bl.addWidget(self.vd_busy_bar)
        self.lbl_vd_elapsed = QtWidgets.QLabel("")   # real time since Generate voice, never an estimate
        self.lbl_vd_elapsed.setAccessibleName("Time elapsed")
        bl.addWidget(self.lbl_vd_elapsed)
        self.lbl_vd_busy_hint = WrapLabel(LOADING_HINT)
        set_role(self.lbl_vd_busy_hint, "muted")
        self.lbl_vd_busy_hint.setVisible(False)   # a short explanation of the stage the backend is really in
        bl.addWidget(self.lbl_vd_busy_hint)
        bl.addStretch(1)
        self.vd_preview.insertWidget(PV_BUSY, busy)

        # ready
        ready = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(ready)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        self.lbl_vd_notice = WrapLabel("")
        self.lbl_vd_notice.setAccessibleName("Sample notice")
        set_role(self.lbl_vd_notice, "notice")
        self.lbl_vd_notice.setVisible(False)
        rl.addWidget(self.lbl_vd_notice)
        notice_row = QtWidgets.QHBoxLayout()
        self.btn_vd_notice_details = QtWidgets.QPushButton("Show details")
        set_role(self.btn_vd_notice_details, "quiet")
        self.btn_vd_notice_details.clicked.connect(self._open_failure_dialog)
        self.btn_vd_notice_details.setVisible(False)
        notice_row.addWidget(self.btn_vd_notice_details)
        notice_row.addStretch(1)
        rl.addLayout(notice_row)
        self.vd_regen_bar = ActivityBar()   # the one indicator while a new sample replaces an old one
        self.vd_regen_bar.setVisible(False)
        rl.addWidget(self.vd_regen_bar)
        text_lbl = QtWidgets.QLabel("Spoken text")
        set_role(text_lbl, "field")
        rl.addWidget(text_lbl)
        self.txt_vd_spoken = text_edit(3, read_only=True)   # the exact words spoken in the audio below
        self.txt_vd_spoken.setTextInteractionFlags(   # select with mouse or keyboard; arrows scroll and move the caret
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.TextSelectableByKeyboard)
        self.txt_vd_spoken.setProperty("role", "reader")   # flat text, not an input: selectable and scrollable
        self.txt_vd_spoken.setFixedHeight(self.txt_vd_spoken.fontMetrics().lineSpacing() * 3 + 8)  # ~3 lines; scrolls
        self.txt_vd_spoken.setAccessibleName("Sample text used for this sample")
        text_lbl.setBuddy(self.txt_vd_spoken)
        rl.addWidget(self.txt_vd_spoken)
        text_row = QtWidgets.QHBoxLayout()
        self.lbl_vd_spoken_note = WrapLabel("Text used for this sample")
        set_role(self.lbl_vd_spoken_note, "muted")
        text_row.addWidget(self.lbl_vd_spoken_note, 1)
        self.btn_vd_edit_text = QtWidgets.QPushButton("Edit text")
        set_role(self.btn_vd_edit_text, "quiet")
        icons.apply(self.btn_vd_edit_text, "pencil")
        self.btn_vd_edit_text.setToolTip("Change the Sample text for the next sample. This audio is not edited.")
        self.btn_vd_edit_text.clicked.connect(self.edit_sample_text)
        text_row.addWidget(self.btn_vd_edit_text)
        rl.addLayout(text_row)

        self.vd_wave = WaveformView()
        self.vd_wave.seek_requested.connect(lambda fraction: self._seek_preview(fraction * self._pv_duration))
        rl.addWidget(self.vd_wave)
        self.sld_vd_seek = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_vd_seek.setAccessibleName("Seek")
        self.sld_vd_seek.setTracking(False)   # dragging only moves the handle; the seek happens on release
        self.sld_vd_seek.setSingleStep(5000)
        self.sld_vd_seek.setPageStep(10000)
        self.sld_vd_seek.setEnabled(False)
        self.sld_vd_seek.valueChanged.connect(self._on_seek_slider)
        self.sld_vd_seek.sliderMoved.connect(lambda ms: self._show_position(ms / 1000.0))
        rl.addWidget(self.sld_vd_seek)
        player = QtWidgets.QHBoxLayout()
        player.setSpacing(8)
        self.lbl_vd_time = QtWidgets.QLabel("0:00")
        self.lbl_vd_time.setAccessibleName("Position")
        set_role(self.lbl_vd_time, "muted")
        player.addWidget(self.lbl_vd_time)
        player.addStretch(1)
        self.btn_vd_back = QtWidgets.QPushButton()
        self.btn_vd_back.setAccessibleName("Back 5 seconds")
        self.btn_vd_back.setToolTip("Back 5 seconds")
        self.btn_vd_back.clicked.connect(lambda: self._seek_preview(self._pv_position() - SEEK_STEP))
        self.btn_vd_play = QtWidgets.QPushButton("Play")
        icons.apply(self.btn_vd_play, "play")
        self.btn_vd_play.setToolTip("Play the generated sample")
        self.btn_vd_play.clicked.connect(self.play_sample)
        self.btn_vd_forward = QtWidgets.QPushButton()
        self.btn_vd_forward.setAccessibleName("Forward 5 seconds")
        self.btn_vd_forward.setToolTip("Forward 5 seconds")
        self.btn_vd_forward.clicked.connect(lambda: self._seek_preview(self._pv_position() + SEEK_STEP))
        for b, glyph in ((self.btn_vd_back, "rotate-ccw"), (self.btn_vd_forward, "rotate-cw")):
            set_role(b, "quiet")
            icons.apply(b, glyph)
        player.addWidget(self.btn_vd_back)
        player.addWidget(self.btn_vd_play)
        player.addWidget(self.btn_vd_forward)
        player.addStretch(1)
        self.lbl_vd_total = QtWidgets.QLabel("0:00")
        self.lbl_vd_total.setAccessibleName("Length")
        set_role(self.lbl_vd_total, "muted")
        player.addWidget(self.lbl_vd_total)
        rl.addLayout(player)
        self.lbl_vd_origin = WrapLabel("")  # which inputs produced this audio
        self.lbl_vd_origin.setAccessibleName("Sample made from")
        set_role(self.lbl_vd_origin, "muted")
        rl.addWidget(self.lbl_vd_origin)
        self.frm_vd_stale = QtWidgets.QFrame()
        set_role(self.frm_vd_stale, "callout")
        callout = QtWidgets.QVBoxLayout(self.frm_vd_stale)
        callout.setContentsMargins(12, 10, 12, 10)
        self.lbl_vd_stale = WrapLabel(
            "Description, language or sample text changed. "
            "Save keeps the original; Generate voice makes a new sample.")
        set_role(self.lbl_vd_stale, "notice")
        callout.addWidget(self.lbl_vd_stale)
        self.frm_vd_stale.setVisible(False)
        self.lbl_vd_stale.setVisible(False)
        rl.addWidget(self.frm_vd_stale)
        rl.addSpacing(4)
        divider = QtWidgets.QFrame()
        set_role(divider, "divider")
        rl.addWidget(divider)
        rl.addSpacing(4)
        name_lbl = QtWidgets.QLabel("Voice name")
        set_role(name_lbl, "field")
        rl.addWidget(name_lbl)
        save_row = QtWidgets.QHBoxLayout()
        save_row.setSpacing(8)
        self.edt_vd_name = QtWidgets.QLineEdit()
        self.edt_vd_name.setPlaceholderText("Name this voice")
        self.edt_vd_name.setAccessibleName("Voice name")
        name_lbl.setBuddy(self.edt_vd_name)
        self.edt_vd_name.textChanged.connect(lambda _t: self._refresh_save_button())
        save_row.addWidget(self.edt_vd_name, 1)
        self.btn_vd_save = QtWidgets.QPushButton("Save")
        icons.apply(self.btn_vd_save, "save")
        self.btn_vd_save.setEnabled(False)
        self.btn_vd_save.clicked.connect(self.save_voice)
        save_row.addWidget(self.btn_vd_save)
        rl.addLayout(save_row)
        self.lbl_vd_name_hint = WrapLabel("")
        set_role(self.lbl_vd_name_hint, "muted")
        self.lbl_vd_name_hint.setVisible(False)
        rl.addWidget(self.lbl_vd_name_hint)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(8)
        self.btn_vd_download = QtWidgets.QPushButton("Download")
        icons.apply(self.btn_vd_download, "download")
        self.btn_vd_download.setToolTip("Export the sample as a WAV file")
        self.btn_vd_download.clicked.connect(self.download_sample)
        self.btn_vd_regenerate = QtWidgets.QPushButton("Regenerate")
        icons.apply(self.btn_vd_regenerate, "refresh-cw")
        self.btn_vd_regenerate.setToolTip("Generate another sample from the current description")
        self.btn_vd_regenerate.clicked.connect(self.generate_sample)
        actions.addWidget(self.btn_vd_download)
        actions.addWidget(self.btn_vd_regenerate)
        actions.addStretch(1)
        rl.addLayout(actions)
        self.vd_preview.insertWidget(PV_READY, ready)

        # error: what failed, what to do, then the technical details on request
        err = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(err)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(8)
        err_title = QtWidgets.QLabel("Couldn't generate the sample")
        set_role(err_title, "error-title")
        el.addWidget(err_title)
        self.lbl_vd_error = WrapLabel("")
        self.lbl_vd_error.setAccessibleName("Error")
        el.addWidget(self.lbl_vd_error)
        self.lbl_vd_error_action = WrapLabel("")
        self.lbl_vd_error_action.setAccessibleName("What you can do")
        set_role(self.lbl_vd_error_action, "muted")
        el.addWidget(self.lbl_vd_error_action)
        err_row = QtWidgets.QHBoxLayout()
        err_row.setSpacing(8)
        self.btn_vd_retry = QtWidgets.QPushButton("Retry")
        self.btn_vd_retry.clicked.connect(self.generate_sample)
        icons.apply(self.btn_vd_retry, "refresh-cw")
        self.btn_vd_error_settings = QtWidgets.QPushButton("Open Settings")
        self.btn_vd_error_settings.clicked.connect(lambda: self.show_page(len(self.nav_buttons) - 1))
        err_row.addWidget(self.btn_vd_retry)
        err_row.addWidget(self.btn_vd_error_settings)
        err_row.addStretch(1)
        el.addLayout(err_row)
        det_row = QtWidgets.QHBoxLayout()
        det_row.setSpacing(8)
        self.btn_vd_details = QtWidgets.QPushButton("Show details")
        self.btn_vd_details.setCheckable(True)
        self.btn_vd_details.setAccessibleName("Technical details")
        self.btn_vd_details.setToolTip("Show the technical details of this failure")
        set_role(self.btn_vd_details, "quiet")
        icons.apply(self.btn_vd_details, "chevron-right")
        self.btn_vd_copy_details = QtWidgets.QPushButton("Copy details")
        self.btn_vd_copy_details.setToolTip("Copy the technical details (private paths and credentials removed)")
        self.btn_vd_copy_details.clicked.connect(self.copy_failure_details)
        set_role(self.btn_vd_copy_details, "quiet")
        icons.apply(self.btn_vd_copy_details, "copy")
        det_row.addWidget(self.btn_vd_details)
        det_row.addWidget(self.btn_vd_copy_details)
        det_row.addStretch(1)
        el.addLayout(det_row)
        # A read-only text box, not a label: it takes keyboard focus, scrolls, and selects with the keyboard.
        self.txt_vd_details = text_edit(4, read_only=True)
        self.txt_vd_details.setAccessibleName("Technical details text")
        self.txt_vd_details.setMaximumHeight(150)
        self.txt_vd_details.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.txt_vd_details.setVisible(False)
        self.btn_vd_details.toggled.connect(self._toggle_error_details)
        el.addWidget(self.txt_vd_details)
        el.addStretch(1)
        self.vd_preview.insertWidget(PV_ERROR, err)
        return panel

    # ---- bottom strip: model / language / advanced
    def _build_options_strip(self):
        strip = QtWidgets.QVBoxLayout()   # Model + Language on one row, Advanced settings beneath: fits the form column
        strip.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(20)
        strip.addLayout(row)

        model_box = QtWidgets.QVBoxLayout()
        model_box.setSpacing(4)
        cap = QtWidgets.QLabel("Model")
        set_role(cap, "caption")
        self.lbl_vd_model = QtWidgets.QLabel(next(n for n, mid in MODELS.items() if mid == MODEL_MAP["voice-design"]))
        self.lbl_vd_model.setToolTip("Voice creation always uses this model; the clone model is loaded when a voice is saved.")
        self.lbl_vd_model.setAccessibleName("Model")
        self.lbl_vd_model.setMinimumHeight(36)  # same height as the language combo beside it
        model_box.addWidget(cap)
        model_box.addWidget(self.lbl_vd_model)
        row.addLayout(model_box)

        lang_box = QtWidgets.QVBoxLayout()
        lang_box.setSpacing(4)
        cap2 = QtWidgets.QLabel("Language")
        set_role(cap2, "caption")
        self.cmb_vd_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        self.cmb_vd_lang.setMinimumWidth(130)
        cap2.setBuddy(self.cmb_vd_lang)
        lang_box.addWidget(cap2)
        lang_box.addWidget(self.cmb_vd_lang)
        row.addLayout(lang_box)

        row.addStretch(1)
        adv_row = QtWidgets.QHBoxLayout()
        strip.addLayout(adv_row)
        self.btn_vd_advanced = QtWidgets.QPushButton("Advanced settings")
        self.btn_vd_advanced.setCheckable(True)
        self.btn_vd_advanced.setAccessibleName("Advanced settings")
        self._set_advanced_icon(False)
        set_role(self.btn_vd_advanced, "quiet")
        self.btn_vd_advanced.toggled.connect(self._toggle_advanced)
        adv_row.addWidget(self.btn_vd_advanced)
        adv_row.addStretch(1)
        return strip

    def _build_advanced_panel(self):
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        seed_lay = QtWidgets.QHBoxLayout()
        seed_lay.setSpacing(8)
        seed_lbl = QtWidgets.QLabel("Seed")
        seed_lbl.setMinimumWidth(48)
        self.cmb_vd_seed_mode, self.spn_vd_seed = seed_row("Random", 42)
        self.lbl_vd_seed_used = QtWidgets.QLabel("")
        seed_lbl.setBuddy(self.cmb_vd_seed_mode)
        for w in (seed_lbl, self.cmb_vd_seed_mode, self.spn_vd_seed):
            seed_lay.addWidget(w)
        seed_lay.addStretch(1)
        lay.addLayout(seed_lay)
        lay.addWidget(self.lbl_vd_seed_used)
        note = QtWidgets.QLabel(
            "Random tries a new voice each time (the seed used is recorded with the saved voice). "
            "Fixed repeats the same sampling. A saved voice is locked via its reference clip, not the seed.")
        note.setWordWrap(True)
        set_role(note, "muted")
        lay.addWidget(note)

        auto_row = QtWidgets.QHBoxLayout()
        auto_row.setSpacing(8)
        self.chk_vd_autosave = QtWidgets.QCheckBox("Auto-save generated voices")
        self.chk_vd_autosave.setToolTip("Automatically save each generated sample as a voice.")
        gender_lbl = QtWidgets.QLabel("Name gender")
        self.cmb_vd_name_gender = QtWidgets.QComboBox()
        self.cmb_vd_name_gender.addItems(["Random", "Male", "Female"])
        self.cmb_vd_name_gender.setToolTip("Gender used for auto-generated voice names.")
        gender_lbl.setBuddy(self.cmb_vd_name_gender)
        self.btn_vd_rand_name = QtWidgets.QPushButton("Suggest name")
        self.btn_vd_rand_name.setToolTip("Fill the voice name with a random, unused name")
        self.btn_vd_rand_name.clicked.connect(self.fill_random_name)
        lay.addWidget(self.chk_vd_autosave)   # the form column is narrow: one control row each
        for w in (gender_lbl, self.cmb_vd_name_gender, self.btn_vd_rand_name):
            auto_row.addWidget(w)
        auto_row.addStretch(1)
        lay.addLayout(auto_row)

        compute = QtWidgets.QHBoxLayout()
        self.lbl_vd_compute = QtWidgets.QLabel("")
        set_role(self.lbl_vd_compute, "muted")
        self.lbl_vd_compute.setAccessibleName("Current compute settings")
        self.btn_vd_open_settings = QtWidgets.QPushButton("Open Settings")
        set_role(self.btn_vd_open_settings, "quiet")
        self.btn_vd_open_settings.clicked.connect(lambda: self.show_page(len(self.nav_buttons) - 1))
        self.lbl_vd_compute.setWordWrap(True)
        lay.addWidget(self.lbl_vd_compute)
        compute.addWidget(self.btn_vd_open_settings)
        compute.addStretch(1)
        lay.addLayout(compute)
        return panel

    def _toggle_advanced(self, on):
        self.vd_advanced.setVisible(on)
        self._set_advanced_icon(on)

    def _set_advanced_icon(self, expanded):
        """The chevron shows the state, and the accessible description says it in words."""
        icons.apply(self.btn_vd_advanced, "chevron-down" if expanded else "chevron-right")
        self.btn_vd_advanced.setAccessibleDescription("Expanded" if expanded else "Collapsed")

    def _update_compute_summary(self):
        device, dtype, attn = self.model_settings()
        self.lbl_vd_compute.setText(f"Device {device} · Dtype {dtype} · Attention {attn}")

    # ---- preview state helpers
    def _set_preview(self, state):
        self.vd_preview.setCurrentIndex(state)
        self.preview_panel.setProperty("empty", state == PV_EMPTY)  # quiet, unfilled panel until audio exists
        style = self.preview_panel.style()
        style.unpolish(self.preview_panel)
        style.polish(self.preview_panel)
        self._refresh_stale_note()

    def _relayout_preview(self):
        # Wrapped labels change height with their text; recompute so nothing is clipped or left oversized.
        self.vd_preview.updateGeometry()
        layout = self.preview_panel.layout()
        layout.invalidate()
        layout.activate()
        self.vd_content.layout().invalidate()

    def _vd_notice(self, text, details=False):
        self.lbl_vd_notice.setText(text)
        self.lbl_vd_notice.setVisible(bool(text))
        self.btn_vd_notice_details.setVisible(bool(text) and details)
        self._relayout_preview()

    def _on_generation_notice(self, message):
        """Backend stage messages: only the model-loading one is reported by the backend, the rest is ours."""
        stage = STAGE_LOADING if message.startswith("Loading ") and message.endswith(" model…") else message
        self._gen_stage = stage
        self.lbl_vd_busy.setText(stage)
        hint = {STAGE_LOADING: LOADING_HINT, STAGE_GENERATING: GENERATING_HINT}.get(stage)
        self.lbl_vd_busy_hint.setText(hint or "")
        self.lbl_vd_busy_hint.setVisible(bool(hint))
        self._tick_elapsed()

    def _generation_elapsed(self):
        return format_duration(max(0.0, time.monotonic() - self._gen_t0)) if self._gen_t0 else ""

    def _tick_elapsed(self):
        """Real elapsed time of the running generation (a monotonic clock), shown beside the stage."""
        if not self._gen_t0:
            return
        elapsed = self._generation_elapsed()
        self.lbl_vd_elapsed.setText(f"{elapsed} elapsed")
        if self.vd_preview.currentIndex() == PV_READY and self.busy:
            self._vd_notice(f"{self._gen_stage} {elapsed} elapsed. Your current sample stays available.")

    def _end_generation_clock(self):
        self._gen_t0 = None
        self.gen_timer.stop()

    def _toggle_error_details(self, on):
        self.txt_vd_details.setVisible(on)
        self.btn_vd_details.setText("Hide details" if on else "Show details")
        icons.apply(self.btn_vd_details, "chevron-down" if on else "chevron-right")
        self._relayout_preview()

    def copy_failure_details(self):
        QtWidgets.QApplication.clipboard().setText(self._failure_details)
        self.status("Technical details copied.")

    def _open_failure_dialog(self):
        """Details for a failed regeneration, where the previous good sample stays on screen."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Technical details")
        dialog.setMinimumSize(560, 320)
        lay = QtWidgets.QVBoxLayout(dialog)
        box = text_edit(10, read_only=True)
        box.setAccessibleName("Technical details text")
        box.setPlainText(self._failure_details)
        lay.addWidget(box, 1)
        buttons = QtWidgets.QDialogButtonBox()
        copy = buttons.addButton("Copy details", QtWidgets.QDialogButtonBox.ActionRole)
        copy.clicked.connect(self.copy_failure_details)
        buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        dialog.exec()

    def _show_ready(self, result):
        wav, sr = np.asarray(result["wav"]), result["sr"]
        self.vd_wave.set_samples(wav)
        self._pv_duration = wav.reshape(-1).size / sr
        self._pv_start = 0.0
        self._pv_last = "stopped"
        self.txt_vd_spoken.setPlainText(result.get("text") or voice_library.SAMPLE_TEXT)
        self.lbl_vd_total.setText(format_duration(self._pv_duration))
        self.sld_vd_seek.setRange(0, int(self._pv_duration * 1000))
        self._refresh_player()
        self.lbl_vd_origin.setText(self._origin_text(result))
        self.lbl_vd_origin.setToolTip(result["description"])
        # Always reachable (tooltip, screen-reader description); shown inline only when it matters (stale).
        self.lbl_vd_preview_title.setToolTip(self.lbl_vd_origin.text())
        self.txt_vd_spoken.setAccessibleDescription(self.lbl_vd_origin.text())
        self._vd_notice("")
        self._set_preview(PV_READY)

    def _show_error(self, message):
        """Explain a failed generation. The cause is named only when it is reliably identifiable;
        otherwise the message stays generic, and the technical details carry the real exception."""
        failure = (getattr(self.task, "failure", None) if self.task else None) or task_errors.Failure.from_message(message)
        explanation = task_errors.classify(failure)
        summary = explanation.summary if explanation else task_errors.UNKNOWN_SUMMARY
        action = explanation.action if explanation else task_errors.UNKNOWN_ACTION
        device, dtype, attn = self.model_settings()
        self._failure_details = task_errors.details_text(failure, context={
            "Model": MODEL_MAP["voice-design"],
            "Settings": f"device={device}, dtype={dtype}, attention={attn}",
            "Language": self.cmb_vd_lang.currentText()})
        self._end_generation_clock()
        self.vd_regen_bar.setVisible(False)
        self.btn_vd_error_settings.setVisible(bool(explanation and explanation.settings))
        if self.pending_voice:  # keep the last good sample available
            self._set_preview(PV_READY)
            self._vd_notice(f"Couldn't generate a new sample. {summary} Your previous sample is still available.",
                            details=True)
            self.status("Couldn't generate a new sample. Your previous sample is still available.")
            return
        self.lbl_vd_error.setText(summary)
        self.lbl_vd_error_action.setText(action)
        self.txt_vd_details.setPlainText(self._failure_details)
        self.btn_vd_details.setChecked(False)
        self._set_preview(PV_ERROR)
        self._announce_error()
        self.status(f"Couldn't generate the sample: {explanation.short}." if explanation
                    else "Couldn't generate the sample. See the Preview for details.")

    def _announce_error(self):
        """Tell assistive technology about the error, and put focus on Retry unless the person is typing."""
        try:
            event = QtGui.QAccessibleEvent(self.lbl_vd_error, QtGui.QAccessible.Event.Alert)
            QtGui.QAccessible.updateAccessibility(event)
        except Exception:  # accessibility must never break error handling
            pass
        focus = QtWidgets.QApplication.focusWidget()
        typing = isinstance(focus, (QtWidgets.QPlainTextEdit, QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox))
        if self.isActiveWindow() and not typing:
            self.btn_vd_retry.setFocus()

    def _on_description_changed(self):
        self.lbl_vd_field_error.setVisible(False)
        self._refresh_stale_note()
        text = self.txt_vd_instruct.toPlainText()
        for chip, name, phrase in self.vd_chips:  # keep chips in step with manual edits
            on = phrase in text
            if chip.isChecked() != on:
                chip.blockSignals(True)
                chip.setChecked(on)
                chip.blockSignals(False)

    @staticmethod
    def _origin_text(sample):
        desc = " ".join(sample["description"].split())
        if len(desc) > 90:
            desc = desc[:87].rstrip() + "…"
        return f"Made from “{desc}” · {sample['language']}"

    def _sample_is_stale(self):
        """True when the description, language or sample text differ from those that produced the sample."""
        pending = self.pending_voice
        return bool(pending) and (
            self.txt_vd_instruct.toPlainText().strip() != pending["description"]
            or self.cmb_vd_lang.currentText() != pending["language"]
            or self._form_sample_text() != self._sample_text_of(pending))

    def _refresh_stale_note(self):
        stale = self._sample_is_stale() and self.vd_preview.currentIndex() == PV_READY
        self.lbl_vd_stale.setVisible(stale)
        self.frm_vd_stale.setVisible(stale)
        self.lbl_vd_origin.setVisible(stale)   # which inputs made this audio: shown when the form has moved on
        self.lbl_vd_preview_title.setText("Previous sample" if stale else "Preview")
        text_changed = bool(self.pending_voice) and self._form_sample_text() != self._sample_text_of(self.pending_voice)
        self.lbl_vd_spoken_note.setText("Text used for this sample (the field has changed)"
                                        if text_changed else "Text used for this sample")
        # The footer must not claim the sample matches the form while it does not.
        if stale and self.lbl_status.text() == READY_STATUS:
            self.status(STALE_STATUS)
        elif not stale and self.lbl_status.text() == STALE_STATUS:
            self.status(READY_STATUS)
        self._refresh_save_button()
        self._relayout_preview()

    def _refresh_save_button(self):
        """Save needs a sample and a usable name; its label says which inputs it will store."""
        name = self.edt_vd_name.text().strip()
        has_sample = bool(self.pending_voice)
        valid = bool(voice_library.slugify(name))
        stale = self._sample_is_stale()
        self.btn_vd_save.setText("Save previous sample" if stale else "Save")
        if stale:
            tip = ("Saves this audio with the description, language and text it was made from "
                   "(shown above), not your current edits.")
        else:
            tip = "Save as a named voice in My Voices"
        if has_sample and not valid:
            tip = "Enter a voice name to enable Save."
        self.btn_vd_save.setToolTip(tip)
        self.btn_vd_save.setEnabled(has_sample and valid and not self.busy)
        if not has_sample or valid:
            hint = ""
        elif not name:
            hint = "Enter a voice name to save."
        else:
            hint = "Use at least one letter or number in the name."
        self.lbl_vd_name_hint.setText(hint)
        self.lbl_vd_name_hint.setVisible(bool(hint))
        self.edt_vd_name.setAccessibleDescription(hint)

    def _toggle_suggestion(self, phrase, on):
        text = self.txt_vd_instruct.toPlainText()
        if on and phrase not in text:
            new = f"{text.rstrip()} {phrase}" if text.strip() else phrase
        elif not on and phrase in text:
            new = " ".join(text.replace(phrase, " ").split())
        else:
            return
        cur = self.txt_vd_instruct.textCursor()  # one undoable edit, so Ctrl+Z restores the old text
        cur.select(QtGui.QTextCursor.Document)
        cur.insertText(new)
        self.txt_vd_instruct.setTextCursor(cur)

    # ---- handlers
    def generate_sample(self):
        description = self.txt_vd_instruct.toPlainText().strip()
        if not description:
            self.status("Error: enter a voice description first.")
            self.lbl_vd_field_error.setVisible(True)
            self.txt_vd_instruct.setFocus()
            return
        text = self._form_sample_text()
        if not text:
            self.status("Error: enter the sample text, or choose Use standard text.")
            self.txt_vd_sample.setFocus()
            return
        language = self.cmb_vd_lang.currentText()
        seed = (self.spn_vd_seed.value() if self.cmb_vd_seed_mode.currentText() == "Fixed"
                else random.randint(0, 2**31 - 1))
        audio_player.stop()
        self._pv_start = 0.0

        def work(task):
            # The same call design_sample makes: it returns at once when the model is already loaded.
            # Making it here lets the app say "Generating" only after the model call has returned.
            self.models.get("voice-design", task.notify.emit)
            task.notify.emit(STAGE_GENERATING)
            wav, sr = self.generator(task).design_sample(description, language, seed, text)
            preview = os.path.join(OUTPUT_DIR, "_voice_preview.wav")
            write_wav(preview, wav, sr)
            return dict(wav=wav, sr=sr, seed=seed, description=description,
                        language=language, preview=preview, text=text)

        def done(result):
            self._end_generation_clock()
            self.vd_regen_bar.setVisible(False)
            self.pending_voice = result
            self.lbl_vd_seed_used.setText(f"Seed used: {result['seed']}")
            self._show_ready(result)
            self.status(READY_STATUS)
            self.play(result["preview"])
            if self.chk_vd_autosave.isChecked():
                gender = self.cmb_vd_name_gender.currentText().lower()
                if gender == "random":
                    gender = random.choice(["male", "female"])
                base = voice_library.random_voice_name(gender)
                name = voice_library.unique_voice_name(base)
                self.edt_vd_name.setText(name)
                self.save_voice(autosaved=True)

        if self.busy:
            self.status("Busy: a task is already running.")
            return
        self._gen_stage = STAGE_PREPARING
        self._gen_t0 = time.monotonic()
        self.gen_timer.start()
        self.lbl_vd_busy.setText(STAGE_PREPARING)
        self.lbl_vd_elapsed.setText("0:00 elapsed")
        self.lbl_vd_busy_hint.setVisible(False)
        if self.pending_voice:  # the current sample stays playable until its replacement is ready
            self._set_preview(PV_READY)
            self._vd_notice(f"{STAGE_PREPARING} 0:00 elapsed. Your current sample stays available.")
            self.vd_regen_bar.setVisible(True)
        else:
            self._set_preview(PV_BUSY)
        # Preview owns the progress and stage text; the footer states the activity once, without animating.
        self.status(FOOTER_CREATING)
        self.run_task(work, done, on_fail=self._show_error, on_notify=self._on_generation_notice,
                      stoppable=False, footer_progress=False, footer_status=False)

    # ---- Preview player: one Play/Pause control over audio_player's own state
    _pv_duration = 0.0   # seconds, from the generated samples themselves
    _pv_start = 0.0      # where Play starts when nothing is playing (set by seeking while stopped)
    _pv_last = "stopped"

    def _form_sample_text(self):
        return self.txt_vd_sample.toPlainText().strip()

    @staticmethod
    def _sample_text_of(sample):
        return sample.get("text") or voice_library.SAMPLE_TEXT

    def _preview_audio_state(self):
        """'playing' / 'paused' / 'stopped' for the Preview's own audio. Audio that belongs to another
        screen (a voice reference, a render) counts as stopped here."""
        pending = self.pending_voice
        if not pending or audio_player.current_path() != pending["preview"]:
            return "stopped"
        return audio_player.state()

    def _pv_position(self):
        if self._preview_audio_state() == "stopped":
            return self._pv_start
        return audio_player.position() or 0.0

    def play_sample(self):
        pending = self.pending_voice
        if not pending:
            self.status("Generate a sample first.")
            return
        state = self._preview_audio_state()
        if state == "playing":
            audio_player.pause() if audio_player.can_control() else audio_player.stop()
        elif state == "paused":
            audio_player.resume()
        else:
            start = self._pv_start if 0 < self._pv_start < self._pv_duration - 0.1 else 0.0
            error = self.play(pending["preview"], start)
            if error:
                self._vd_notice(f"Couldn't play the sample: {error}")
        self._refresh_audio_state()
        self._refresh_player()

    def _seek_preview(self, seconds):
        if not self.pending_voice or self._pv_duration <= 0 or not audio_player.can_control():
            return
        seconds = min(max(0.0, seconds), max(0.0, self._pv_duration - 0.05))
        if self._preview_audio_state() == "stopped":
            self._pv_start = seconds
        else:
            audio_player.seek(seconds)
        self._refresh_player()

    def _on_seek_slider(self, ms):
        if not self._pv_syncing:
            self._seek_preview(ms / 1000.0)

    _pv_syncing = False

    def _show_position(self, seconds):
        self.lbl_vd_time.setText(format_duration(seconds))
        self.sld_vd_seek.setAccessibleDescription(
            f"{format_duration(seconds)} of {format_duration(self._pv_duration)}")

    def _refresh_player(self):
        """Mirror the real playback state into the Preview. Called by the timer and after every action."""
        if not self.pending_voice:
            return
        state = self._preview_audio_state()
        if state == "stopped" and self._pv_last in ("playing", "paused"):
            self._pv_start = 0.0   # finished or stopped: the next Play starts from the beginning
        self._pv_last = state
        control = audio_player.can_control()
        playing = state == "playing"
        label = ("Pause" if control else "Stop") if playing else "Play"
        if self.btn_vd_play.text() != label:
            self.btn_vd_play.setText(label)
            icons.apply(self.btn_vd_play, ("pause" if control else "square") if playing else "play")
            self.btn_vd_play.setToolTip({"Pause": "Pause playback", "Stop": "Stop playback"}.get(
                label, "Resume the sample" if state == "paused" else "Play the generated sample"))
        if state == "paused":
            self.btn_vd_play.setToolTip("Resume playback")
        for control_widget in (self.sld_vd_seek, self.btn_vd_back, self.btn_vd_forward):
            control_widget.setEnabled(control)
        position = min(self._pv_position(), self._pv_duration)
        if not self.sld_vd_seek.isSliderDown():
            self._pv_syncing = True
            self.sld_vd_seek.setValue(int(position * 1000))
            self._pv_syncing = False
            self._show_position(position)
        if not control:
            self.lbl_vd_time.setText("–")
        active = state != "stopped" or self._pv_start > 0
        self.vd_wave.set_progress(position / self._pv_duration if active and self._pv_duration else None)

    def _tick_player(self):
        if self.pending_voice and self.vd_preview.currentIndex() == PV_READY and self.isVisible():
            self._refresh_player()
            self._refresh_audio_state()   # the footer reads the same player state in the same tick

    def edit_sample_text(self):
        """Take the person to the Sample text field (the existing audio is never edited)."""
        self.txt_vd_sample.setFocus()
        QtCore.QTimer.singleShot(
            0, lambda: self.stack.widget(PAGE_CREATE).ensureWidgetVisible(self.txt_vd_sample))

    def _on_sample_text_changed(self):
        self._refresh_stale_note()

    def download_sample(self):
        pending = self.pending_voice
        if not pending:
            self.status("Generate a sample first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Download sample", "voice-sample.wav", "WAV audio (*.wav)")
        if not path:
            return
        try:
            write_wav(path, np.asarray(pending["wav"]), pending["sr"])
        except Exception as e:
            self._vd_notice(f"Couldn't save the file: {e}")
            return
        self._vd_notice(f"Saved a copy to {path}")
        self.status(f"Sample exported to: {path}")

    def save_voice(self, autosaved=False):
        pending = self.pending_voice
        name = self.edt_vd_name.text().strip()
        if not pending:
            self.status("Error: generate and audition a sample first.")
            return
        if not voice_library.slugify(name):
            self.status("Error: enter a voice name.")
            self._vd_notice("Enter a voice name to save.")
            self.edt_vd_name.setFocus()
            return
        was_stale = self._sample_is_stale()  # the snapshot below keeps the sample's own inputs either way
        overwrite = False
        if voice_library.voice_exists(name):
            if autosaved:
                name = voice_library.unique_voice_name(name)
                self.edt_vd_name.setText(name)
            else:
                answer = QtWidgets.QMessageBox.question(
                    self, "Overwrite voice?", f"A voice named '{name}' already exists. Replace it?")
                if answer != QtWidgets.QMessageBox.Yes:
                    return
                overwrite = True

        def work(task):
            text = self._sample_text_of(pending)   # the words actually spoken in this audio
            prompt = self.generator(task).build_voice_prompt(pending["wav"], pending["sr"], text)
            voice_library.save_voice(
                name, pending["wav"], pending["sr"], prompt, pending["description"],
                pending["language"], pending["seed"],
                model_ids=dict(MODEL_MAP),
                overwrite=overwrite,
                sample_text=text,
            )
            return name

        def done(saved_name):
            self.refresh_voice_lists()
            self.status(f"Voice '{saved_name}' saved to voices/{voice_library.slugify(saved_name)}/")
            note = f"Saved as '{saved_name}'. Find it in My Voices."
            if was_stale:
                note = f"Saved as '{saved_name}' with the description it was made from. Find it in My Voices."
            self._vd_notice(note)

        self._vd_notice("")
        self.run_task(work, done, on_fail=lambda m: self._vd_notice(f"Couldn't save the voice: {m}"),
                      stoppable=False)

    def fill_random_name(self):
        gender = self.cmb_vd_name_gender.currentText().lower()
        if gender == "random":
            gender = random.choice(["male", "female"])
        name = voice_library.random_voice_name(gender)
        name = voice_library.unique_voice_name(name)
        self.edt_vd_name.setText(name)
        self.status(f"Suggested name: {name}")

    def refresh_voice_lists(self):
        names = voice_library.list_voices()
        self.cmb_vc_voice.clear()
        self.cmb_vc_voice.addItems([""] + names)
        if hasattr(self, "audiobook_tab"):
            self.audiobook_tab.refresh_voices()

    # ------------------------------------------------------------------ VoiceClone
    def _build_clone_tab(self):
        page, lay, bar = tab_page(None)
        lay.addWidget(page_header("Clone Voice"))

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Saved voice:"))
        self.cmb_vc_voice = QtWidgets.QComboBox()
        self.cmb_vc_voice.setMinimumWidth(240)
        self.cmb_vc_voice.currentTextChanged.connect(self.on_clone_voice_changed)
        row.addWidget(self.cmb_vc_voice)
        btn = QtWidgets.QPushButton("Refresh")
        btn.clicked.connect(self.refresh_voice_lists)
        row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addWidget(QtWidgets.QLabel("Text to synthesize:"))
        self.txt_vc_text = text_edit(10, text="Hello, I am cloning this voice.")
        lay.addWidget(self.txt_vc_text, 1)
        lay.addWidget(DirectionBar(self.txt_vc_text))
        help_lbl = QtWidgets.QLabel(
            "The Base (clone) model takes no instructions, so each [tag] is performed by generating a short "
            "emotional reference with VoiceDesign (voice description + tag), anchored to your voice's speaker "
            "embedding, and cloning from it. Identity stays close but not identical to the neutral voice; "
            "audition emotional passages. " + DIRECTION_HELP +
            " The Voice Description box is used only when [emotion] tags are present; it is combined with each "
            "tag to form the style instruct. Without tags, the box has no effect.")
        help_lbl.setWordWrap(True)
        set_role(help_lbl, "muted")
        lay.addWidget(help_lbl)

        form = QtWidgets.QFormLayout()
        self.cmb_vc_lang = combo(LANGUAGES, DEFAULT_LANGUAGE)
        form.addRow("Language:", self.cmb_vc_lang)
        self.cmb_vc_seed_mode, self.spn_vc_seed = seed_row("Fixed", 42)
        self.spn_vc_temp = QtWidgets.QDoubleSpinBox()
        self.spn_vc_temp.setRange(0.1, 1.5)
        self.spn_vc_temp.setSingleStep(0.05)
        self.spn_vc_temp.setValue(0.7)
        self.spn_vc_temp.setToolTip("Lower = steadier voice, flatter delivery. Library default is 0.9.")
        r = QtWidgets.QHBoxLayout()
        for w in (self.cmb_vc_seed_mode, self.spn_vc_seed, QtWidgets.QLabel("Temperature:"), self.spn_vc_temp):
            r.addWidget(w)
        r.addStretch(1)
        form.addRow("Seed:", r)
        lay.addLayout(form)

        lay.addWidget(QtWidgets.QLabel("Voice description (used for [tag] direction; filled from a saved voice):"))
        self.txt_vc_instruct = text_edit(3, "e.g. Warm, gentle female voice, calm and soothing tone.")
        lay.addWidget(self.txt_vc_instruct)

        self.grp_ref = QtWidgets.QGroupBox("Or clone from reference audio (ignored when a saved voice is selected)")
        g = QtWidgets.QFormLayout(self.grp_ref)
        self.edt_vc_ref = QtWidgets.QLineEdit()
        b1 = QtWidgets.QPushButton("Browse…")
        b1.clicked.connect(lambda: self.browse(self.edt_vc_ref, "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)"))
        g.addRow("Reference audio:", self.row_with(self.edt_vc_ref, b1))
        self.txt_vc_ref_text = text_edit(3, "Exact transcript of the reference audio")
        g.addRow("Transcript:", self.txt_vc_ref_text)
        self.chk_vc_xvec = QtWidgets.QCheckBox("X-vector only (speaker embedding only, no transcript needed)")
        g.addRow("", self.chk_vc_xvec)
        self.edt_vc_reuse = QtWidgets.QLineEdit()
        b2 = QtWidgets.QPushButton("Browse…")
        b2.clicked.connect(lambda: self.browse(self.edt_vc_reuse, "NPZ (*.npz)"))
        g.addRow("Reuse prompt (NPZ):", self.row_with(self.edt_vc_reuse, b2))
        self.edt_vc_save = QtWidgets.QLineEdit()
        b3 = QtWidgets.QPushButton("Browse…")
        b3.clicked.connect(lambda: self.browse(self.edt_vc_save, "NPZ (*.npz)", save=True))
        g.addRow("Save prompt to (NPZ):", self.row_with(self.edt_vc_save, b3))

        self.edt_vc_new_name = QtWidgets.QLineEdit()
        self.edt_vc_new_name.setPlaceholderText("New voice name")
        btn_vc_new = QtWidgets.QPushButton("Save as voice")
        btn_vc_new.clicked.connect(self.save_clone_as_voice)
        g.addRow("Save this clone as a voice:", self.row_with(self.edt_vc_new_name, btn_vc_new))
        lay.addWidget(self.grp_ref)

        self.btn_vc_generate = QtWidgets.QPushButton("Generate")
        set_role(self.btn_vc_generate, "primary")
        self.btn_vc_generate.clicked.connect(self.generate_clone)
        bar.addWidget(self.btn_vc_generate)
        bar.addStretch(1)
        return page

    @staticmethod
    def row_with(edit, button):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(edit, 1)
        lay.addWidget(button)
        return w

    def browse(self, edit, flt, save=False):
        fn = QtWidgets.QFileDialog.getSaveFileName if save else QtWidgets.QFileDialog.getOpenFileName
        path, _ = fn(self, "Select file", "", flt)
        if path:
            edit.setText(path)

    def on_clone_voice_changed(self, name):
        self.grp_ref.setEnabled(not name)
        if name:
            try:
                cfg, _, _ = voice_library.load_voice(name)
            except Exception as e:
                self.status(f"Could not load voice: {e}")
                return
            self.txt_vc_instruct.setPlainText(cfg["description"])
            self.cmb_vc_lang.setCurrentText(cfg["language"])
            self.status(f"Using saved voice '{name}'.")

    def generate_clone(self):
        text = self.txt_vc_text.toPlainText().strip()
        if not text:
            self.status("Error: no text provided.")
            return
        args = dict(
            text=text, language=self.cmb_vc_lang.currentText(),
            min_words=self.spn_min.value(), max_words=self.spn_max.value(),
            seed=self.spn_vc_seed.value() if self.cmb_vc_seed_mode.currentText() == "Fixed" else None,
            temperature=self.spn_vc_temp.value(),
            voice_name=self.cmb_vc_voice.currentText() or None,
            ref_audio=self.edt_vc_ref.text().strip(),
            ref_text=self.txt_vc_ref_text.toPlainText().strip() or None,
            x_vector_only=self.chk_vc_xvec.isChecked(),
            reuse_prompt=self.edt_vc_reuse.text().strip() or None,
            save_prompt=self.edt_vc_save.text().strip() or None,
            description=self.txt_vc_instruct.toPlainText(),
        )
        self.run_task(lambda t: self.generator(t).clone(**args), self.finish_long_form("VoiceClone"))

    def save_clone_as_voice(self):
        """Build a clone prompt from reference audio and save it as a named voice."""
        name = self.edt_vc_new_name.text().strip()
        if not voice_library.slugify(name):
            self.status("Error: enter a voice name.")
            return
        ref_audio = self.edt_vc_ref.text().strip()
        if not ref_audio:
            self.status("Error: provide reference audio first.")
            return
        ref_text = self.txt_vc_ref_text.toPlainText().strip() or None
        x_vector_only = self.chk_vc_xvec.isChecked()
        language = self.cmb_vc_lang.currentText()
        description = self.txt_vc_instruct.toPlainText().strip()
        seed = self.spn_vc_seed.value() if self.cmb_vc_seed_mode.currentText() == "Fixed" else 0

        def work(task):
            wav, sr = load_wav(ref_audio)
            prompt = self.generator(task).build_clone_prompt(
                ref_audio=(wav, sr), ref_text=ref_text, x_vector_only=x_vector_only)
            overwrite = voice_library.voice_exists(name)
            voice_library.save_voice(
                name, wav, sr, prompt, description, language, seed,
                model_ids=dict(MODEL_MAP), overwrite=overwrite)
            return name

        def done(saved_name):
            self.refresh_voice_lists()
            self.cmb_vc_voice.setCurrentText(saved_name)
            self.status(f"Voice '{saved_name}' saved to voices/{voice_library.slugify(saved_name)}/")

        self.run_task(work, done, stoppable=False)

    # ------------------------------------------------------------------ Voices tab
    def _build_voices_tab(self):
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        page.setWidget(body)
        frame = QtWidgets.QHBoxLayout(body)
        frame.setContentsMargins(32, 32, 32, 24)
        content = QtWidgets.QWidget()
        content.setMaximumWidth(CONTENT_MAX_WIDTH)
        frame.addWidget(content, 100)
        frame.addStretch(1)
        outer = QtWidgets.QVBoxLayout(content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(16)

        # Header: title on the left, the two ways to add a voice on the right (neutral, not primary).
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(page_header(
            "My Voices", "Saved voices are shared by Clone Voice and Audiobooks."), 1, QtCore.Qt.AlignTop)
        self.btn_voice_import = QtWidgets.QPushButton("Import voice…")
        self.btn_voice_import.setToolTip("Import a voice folder (must contain reference.wav + voice.json).")
        self.btn_voice_import.clicked.connect(self.import_voice)
        icons.apply(self.btn_voice_import, "download")
        self.btn_voice_create = QtWidgets.QPushButton("Create from audio…")
        self.btn_voice_create.setToolTip("Build a new voice from a WAV file + transcript.")
        self.btn_voice_create.clicked.connect(self.create_voice_from_audio)
        icons.apply(self.btn_voice_create, "wand-sparkles")
        for b in (self.btn_voice_import, self.btn_voice_create):
            head.addWidget(b, 0, QtCore.Qt.AlignTop)
        outer.addLayout(head)

        # List toolbar: the count is context for the list itself (the footer stays for action feedback).
        bar = QtWidgets.QHBoxLayout()
        self.lbl_voice_count = QtWidgets.QLabel("")
        set_role(self.lbl_voice_count, "muted")
        bar.addWidget(self.lbl_voice_count)
        bar.addStretch(1)
        self.btn_voice_refresh = QtWidgets.QPushButton("Refresh")
        set_role(self.btn_voice_refresh, "quiet")
        self.btn_voice_refresh.setToolTip("Re-scan the voices folder.")
        self.btn_voice_refresh.clicked.connect(self.refresh_voices_clicked)
        icons.apply(self.btn_voice_refresh, "refresh-cw")
        bar.addWidget(self.btn_voice_refresh)
        outer.addLayout(bar)

        # The list (native QListWidget, rows painted by a delegate) or, when empty, an explanation.
        self.lst_voices = voice_list.VoiceList()
        set_role(self.lst_voices, "voices")
        self.lst_voices.setItemDelegate(voice_list.VoiceRowDelegate(self.lst_voices))
        self.lst_voices.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.lst_voices.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lst_voices.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.lst_voices.setMouseTracking(True)
        self.lst_voices.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lst_voices.setUniformItemSizes(True)
        self.lst_voices.setMinimumHeight(180)
        self.lst_voices.setAccessibleName("Saved voices")
        self.lst_voices.setAccessibleDescription(
            "Use the arrow keys to move, Shift or Command with the arrow keys to select several. "
            "Return plays the reference audio.")
        outer.addWidget(self.lst_voices, 1)

        self.frm_voices_empty = QtWidgets.QFrame()
        set_role(self.frm_voices_empty, "callout")
        empty = QtWidgets.QVBoxLayout(self.frm_voices_empty)
        empty.setContentsMargins(24, 24, 24, 24)
        empty.setSpacing(8)
        empty_title = QtWidgets.QLabel("No saved voices yet")
        set_role(empty_title, "heading")
        self.lbl_voices_empty = QtWidgets.QLabel(
            "Create a voice from a description or a recording. Saved voices appear here.")
        self.lbl_voices_empty.setWordWrap(True)
        set_role(self.lbl_voices_empty, "muted")
        self.btn_voice_go_create = QtWidgets.QPushButton("Go to Create Voice")
        self.btn_voice_go_create.clicked.connect(self._go_to_create_voice)
        icons.apply(self.btn_voice_go_create, "wand-sparkles")
        empty.addWidget(empty_title)
        empty.addWidget(self.lbl_voices_empty)
        empty.addSpacing(8)
        empty.addWidget(self.btn_voice_go_create, 0, QtCore.Qt.AlignLeft)
        outer.addWidget(self.frm_voices_empty, 0, QtCore.Qt.AlignTop)

        # Selected-voice bar: names the selection and offers what applies to it.
        self.frm_voice_actions = QtWidgets.QFrame()
        set_role(self.frm_voice_actions, "callout")
        sel = QtWidgets.QHBoxLayout(self.frm_voice_actions)
        sel.setContentsMargins(16, 10, 16, 10)
        sel.setSpacing(8)
        self.lbl_voice_selection = QtWidgets.QLabel("")
        set_role(self.lbl_voice_selection, "heading")
        self.lbl_voice_selection.setMinimumWidth(0)
        self.lbl_voice_selection.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sel.addWidget(self.lbl_voice_selection, 1)
        self.btn_voice_play = QtWidgets.QPushButton("Play reference")
        self.btn_voice_play.clicked.connect(self.play_voice_tab)
        icons.apply(self.btn_voice_play, "play")
        self.btn_voice_rename = QtWidgets.QPushButton("Rename")
        self.btn_voice_rename.clicked.connect(self.rename_voice_tab)
        self.btn_voice_delete = QtWidgets.QPushButton("Delete")
        self.btn_voice_delete.clicked.connect(self.delete_voice_tab)
        act_transcript = QtGui.QAction("Reference transcript…", self)
        act_transcript.triggered.connect(self.edit_csm_transcript)
        act_f5 = QtGui.QAction("F5 reference…", self)
        act_f5.triggered.connect(self.edit_f5_reference)
        # Engine-specific references are rare and need exactly one voice, so they live in a menu.
        self.btn_voice_engine_refs = QtWidgets.QPushButton("Engine references")
        set_role(self.btn_voice_engine_refs, "menu")
        engine_menu = QtWidgets.QMenu(self.btn_voice_engine_refs)
        engine_menu.addAction(act_transcript)
        engine_menu.addAction(act_f5)
        self.btn_voice_engine_refs.setMenu(engine_menu)
        self.act_voice_transcript, self.act_voice_f5 = act_transcript, act_f5
        for b in (self.btn_voice_play, self.btn_voice_rename, self.btn_voice_delete, self.btn_voice_engine_refs):
            sel.addWidget(b)
        outer.addWidget(self.frm_voice_actions)
        outer.addStretch(0)   # takes the spare height only while the library is empty
        self._voices_outer = outer

        self.lst_voices.itemSelectionChanged.connect(self._refresh_voice_actions)
        self.lst_voices.itemActivated.connect(lambda item: self.play_voice_by_slug(item.data(voice_list.SLUG_ROLE)))
        self._fill_voices_list()  # silent: a library count is not the status of the Create Voice task
        return page

    def _go_to_create_voice(self):
        self.show_page(PAGE_CREATE)
        self.txt_vd_instruct.setFocus()   # the next thing the person does is describe a voice

    def refresh_voices_tab(self):
        """Re-read the library. Silent: callers report their own result in the footer."""
        self._fill_voices_list()

    def refresh_voices_clicked(self):
        self._fill_voices_list()
        self.status("Voice list refreshed.")

    def _fill_voices_list(self, select=None):
        """Rebuild the list; keep the current selection (or select `select`, a list of slugs)."""
        keep = set(select if select is not None else self.selected_voice_slugs())
        details = voice_library.list_voice_details()
        self.lst_voices.blockSignals(True)
        self.lst_voices.clear()
        for detail in details:
            item = voice_list.make_item(detail)
            self.lst_voices.addItem(item)
            if detail["slug"] in keep:
                item.setSelected(True)
        self.lst_voices.blockSignals(False)
        if keep and self.lst_voices.selectedItems():
            self.lst_voices.setCurrentItem(self.lst_voices.selectedItems()[0], QtCore.QItemSelectionModel.NoUpdate)
        self.lst_voices.setVisible(bool(details))
        self._voices_outer.setStretchFactor(self.lst_voices, 1 if details else 0)
        self._voices_outer.setStretch(self._voices_outer.count() - 1, 0 if details else 1)
        self.frm_voices_empty.setVisible(not details)
        self.frm_voice_actions.setVisible(bool(details))
        self.lbl_voice_count.setText(
            "" if not details else "1 voice" if len(details) == 1 else f"{len(details)} voices")
        self._refresh_voice_actions()
        return len(details)

    def _refresh_voice_actions(self):
        """Name the selection and enable only what applies: one voice for play, rename and
        references; one or more for delete. Handlers keep their own error fallback."""
        slugs = self.selected_voice_slugs()
        names = self.selected_voice_names()
        count = len(slugs)
        one = count == 1
        if count == 0:
            text, tip = "No voice selected", ""
            self.lbl_voice_selection.setProperty("role", "muted")
        else:
            text = names[0] if one else f"{count} voices selected"
            tip = names[0] if one else ", ".join(names)
            self.lbl_voice_selection.setProperty("role", "heading")
        self.lbl_voice_selection.style().unpolish(self.lbl_voice_selection)
        self.lbl_voice_selection.style().polish(self.lbl_voice_selection)
        metrics = self.lbl_voice_selection.fontMetrics()
        self.lbl_voice_selection.setText(metrics.elidedText(text, QtCore.Qt.ElideRight, 260))
        self.lbl_voice_selection.setToolTip(tip)
        self.lbl_voice_selection.setAccessibleName(text)

        need_one = "Select exactly one voice."
        for button, enabled_tip, accessible in (
                (self.btn_voice_play, "Play the selected voice's reference audio. "
                                      "Return on a row also plays it.", f"Play reference for {names[0] if one else ''}"),
                (self.btn_voice_rename, "Rename the selected voice.", f"Rename {names[0] if one else ''}"),
                (self.btn_voice_engine_refs,
                 "Add a reference transcript or F5 reference for the selected voice.", "")):
            button.setEnabled(one)
            button.setToolTip(enabled_tip if one else need_one)
            button.setAccessibleName(accessible.strip() if one else "")
        self.btn_voice_delete.setEnabled(count >= 1)
        self.btn_voice_delete.setToolTip(
            "Permanently delete the selected voice(s). You will be asked to confirm." if count
            else "Select one or more voices to delete.")
        self.btn_voice_delete.setAccessibleName(
            "" if not count else f"Delete {names[0]}" if one else f"Delete {count} voices")

    def selected_voice_slugs(self):
        return sorted({item.data(voice_list.SLUG_ROLE) for item in self.lst_voices.selectedItems()})

    def selected_voice_names(self):
        by_slug = {item.data(voice_list.SLUG_ROLE): item.data(voice_list.NAME_ROLE)
                   for item in self.lst_voices.selectedItems()}
        return [by_slug[s] for s in sorted(by_slug)]

    def selected_voice_slug(self):
        slugs = self.selected_voice_slugs()
        return slugs[0] if len(slugs) == 1 else None

    def rename_voice_tab(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice to rename.")
            return
        current = self.selected_voice_names()[0]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename voice", f"New name for '{current}':", text=current)
        if not ok or not name.strip():
            return
        try:
            voice_library.rename_voice(slug, name.strip())
            self._fill_voices_list(select=[voice_library.slugify(name.strip())])
            self.refresh_voice_lists()
            self.status(f"Renamed '{current}' to '{name.strip()}'.")
        except Exception as e:
            self.status(f"Rename failed: {e}")

    def delete_voice_tab(self):
        slugs = self.selected_voice_slugs()
        if not slugs:
            return
        names = self.selected_voice_names()
        shown = ", ".join(names[:5]) + (f" and {len(names) - 5} more" if len(names) > 5 else "")
        answer = QtWidgets.QMessageBox.question(
            self, "Delete voices?",
            f"Delete {len(slugs)} voice(s): {shown}? This cannot be undone.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for slug in slugs:
            voice_library.delete_voice(slug)
        self._fill_voices_list(select=[])
        self.refresh_voice_lists()
        self.status(f"Deleted {len(slugs)} voice(s).")

    def play_voice_tab(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice to play.")
            return
        self.play_voice_by_slug(slug)

    def edit_csm_transcript(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice.")
            return
        try:
            with open(os.path.join(voice_library.voice_dir(slug), "voice.json"), encoding="utf-8") as voice_file:
                config = json.load(voice_file)
            text, accepted = QtWidgets.QInputDialog.getMultiLineText(
                self, "Reference transcript",
                "Enter the exact words spoken in the entire reference recording:",
                config.get("csm_ref_text", config.get("ref_text", "")))
            if accepted:
                voice_library.set_csm_reference_text(slug, text)
                self.status(f"Reference transcript saved for '{slug}'.")
        except (OSError, ValueError) as exc:
            self.status(f"Error saving reference transcript: {exc}")

    def edit_f5_reference(self):
        slug = self.selected_voice_slug()
        if slug is None:
            self.status("Error: select exactly one voice.")
            return
        source, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select F5 reference (12 seconds or less)", "", "WAV audio (*.wav)")
        if not source:
            return
        text, accepted = QtWidgets.QInputDialog.getMultiLineText(
            self, "F5 reference transcript", "Enter the exact words spoken in this recording:")
        if not accepted:
            return
        try:
            voice_library.set_f5_reference(slug, source, text)
            self.status(f"F5 reference saved for '{slug}'.")
        except (OSError, ValueError) as exc:
            self.status(f"Error saving F5 reference: {exc}")

    def play_voice_by_slug(self, slug):
        path = os.path.join(voice_library.voice_dir(slug), "reference.wav")
        if not os.path.isfile(path):
            self.status(f"Error: reference.wav not found for '{slug}'.")
            return
        self.play(path)

    def import_voice(self):
        src = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select voice folder to import", voice_library.VOICES_DIR)
        if not src:
            return
        cfg_path = os.path.join(src, "voice.json")
        wav_path = os.path.join(src, "reference.wav")
        if not os.path.isfile(cfg_path):
            self.status("Error: selected folder has no voice.json.")
            return
        if not os.path.isfile(wav_path):
            self.status("Error: selected folder has no reference.wav.")
            return
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            self.status(f"Error reading voice.json: {e}")
            return
        name = cfg.get("name", os.path.basename(src))
        if not voice_library.slugify(name):
            self.status("Error: voice name is invalid.")
            return
        name = voice_library.unique_voice_name(name)
        dst = voice_library.voice_dir(name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        # Ensure voice.json name matches the new folder name
        cfg["name"] = name
        with open(os.path.join(dst, "voice.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        self.refresh_voices_tab()
        self.refresh_voice_lists()
        self.status(f"Imported voice '{name}'.")

    def create_voice_from_audio(self):
        """Build a new voice from a WAV file + transcript (no model generation needed for the prompt)."""
        wav_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select reference audio", "", "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)")
        if not wav_path:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New voice name", "Voice name:", text=os.path.splitext(os.path.basename(wav_path))[0])
        if not ok or not voice_library.slugify(name.strip()):
            self.status("Error: enter a valid voice name.")
            return
        name = voice_library.unique_voice_name(name.strip())
        language, ok2 = QtWidgets.QInputDialog.getItem(
            self, "Language", "Language:", LANGUAGES, 0, False)
        if not ok2:
            return
        description, ok3 = QtWidgets.QInputDialog.getMultiLineText(
            self, "Voice description", "Describe the voice (used for [emotion] tags):", "")
        if not ok3:
            return
        seed, ok4 = QtWidgets.QInputDialog.getInt(
            self, "Seed", "Seed (0 = random):", 0, 0, 2**31 - 1)
        if not ok4:
            return
        if seed == 0:
            seed = random.randint(0, 2**31 - 1)

        def work(task):
            wav, sr = load_wav(wav_path)
            prompt = self.generator(task).build_clone_prompt(
                ref_audio=(wav, sr), ref_text=None, x_vector_only=True)
            voice_library.save_voice(
                name, wav, sr, prompt, description, language, seed,
                model_ids=dict(MODEL_MAP), overwrite=False)
            return name

        def done(saved_name):
            self.refresh_voices_tab()
            self.refresh_voice_lists()
            self.status(f"Voice '{saved_name}' created from audio.")

        self.run_task(work, done, stoppable=False)

    @property
    def action_buttons(self):
        return (self.btn_vd_generate, self.btn_vd_save, self.btn_vd_regenerate, self.btn_vd_retry,
                self.btn_vc_generate, self.audiobook_tab.btn_render, self.audiobook_tab.btn_preview)


# ------------------------------------------------------------------ Voice Manager dialog
class VoiceManagerDialog(QtWidgets.QDialog):
    """View, rename, and delete all saved voices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Manager")
        self.setMinimumWidth(700)
        self.setMinimumHeight(400)

        lay = QtWidgets.QVBoxLayout(self)

        lay.addWidget(QtWidgets.QLabel("All saved voices (stored in voices/<name>/):"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Description", "Language", "Seed", "Created"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for c in (0, 2, 3, 4):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        lay.addWidget(self.table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_rename = QtWidgets.QPushButton("Rename selected")
        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        for b in (btn_refresh, self.btn_rename, self.btn_delete, btn_close):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self.refresh()

    def refresh(self):
        details = voice_library.list_voice_details()
        self.table.setRowCount(len(details))
        for r, d in enumerate(details):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(d["name"]))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(d["description"]))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(d["language"]))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(d["seed"])))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(d["created"]))
            # store slug for operations
            self.table.item(r, 0).setData(QtCore.Qt.UserRole, d["slug"])

    def selected_slugs(self):
        slugs = []
        for item in self.table.selectedItems():
            if item.column() == 0:
                slugs.append(item.data(QtCore.Qt.UserRole))
        return sorted(set(slugs))

    def rename_selected(self):
        slugs = self.selected_slugs()
        if len(slugs) != 1:
            QtWidgets.QMessageBox.information(
                self, "Rename voice", "Select exactly one voice to rename.")
            return
        slug = slugs[0]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename voice", f"New name for '{slug}':", text=slug)
        if not ok or not name.strip():
            return
        try:
            voice_library.rename_voice(slug, name.strip())
            self.refresh()
            if self.parent():
                self.parent().refresh_voice_lists()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename failed", str(e))

    def delete_selected(self):
        slugs = self.selected_slugs()
        if not slugs:
            return
        names = ", ".join(slugs)
        answer = QtWidgets.QMessageBox.question(
            self, "Delete voices?",
            f"Delete {len(slugs)} voice(s): {names}? This cannot be undone.")
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for slug in slugs:
            voice_library.delete_voice(slug)
        self.refresh()
        if self.parent():
            self.parent().refresh_voice_lists()


def configure_keyboard_navigation(app):
    """Let Tab reach every control on every platform.

    macOS only Tab-focuses text and list controls unless the system's Full Keyboard Access is on.
    Then the description box is the only Tab stop, Qt finds nowhere to move and inserts a tab
    character instead (a keyboard trap). The app promises full keyboard operation, so it opts in.
    """
    app.styleHints().setTabFocusBehavior(QtCore.Qt.TabFocusAllControls)


def main():
    app = QtWidgets.QApplication(sys.argv)
    configure_keyboard_navigation(app)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
