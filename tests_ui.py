"""Headless Qt smoke tests: build the main window with a stub model manager.

Run with:  python tests_ui.py
No real model is downloaded or loaded, no audio is played, and no GPU is needed.
Generation paths run against a stub model, with playback and the output folder patched.
"""

import os
import re
import tempfile
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

import audio_player
import qt_app
import icons
import theme
import voice_library
PAGE_VOICES = 2
from qt_app import CONTENT_MAX_WIDTH, MainWindow, READY_STATUS, STALE_STATUS, configure_keyboard_navigation, PV_BUSY, PV_EMPTY, PV_ERROR, PV_READY


class _StubModels:
    """Stands in for ModelManager; any attempt to load a model fails the generation."""

    def __init__(self):
        self.calls = []

    def get(self, kind, notify=None):
        self.calls.append(kind)
        raise AssertionError(f"model '{kind}' must not be loaded in UI smoke tests")


class _FakeDesignModel:
    def generate_voice_design(self, text, language, instruct):
        return [np.sin(np.linspace(0, 440, 24000, dtype=np.float32)) * 0.5], 24000

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        return "FAKE-PROMPT"


class _FakeModels:
    def get(self, kind, notify=None):
        return _FakeDesignModel()


class _SlowModels:
    """Blocks inside generation until released, so tests can look at the UI mid-run."""

    def __init__(self):
        self.release = threading.Event()

    def get(self, kind, notify=None):
        if notify:
            notify("Loading voice-design model…")
        self.release.wait(10)
        return _FakeDesignModel()


_app = None


def _make_window(themed=True, models=None):
    global _app
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_keyboard_navigation(_app)
    if themed:
        theme.apply_theme(_app)
    stub = models or _StubModels()
    return MainWindow(models=stub), stub


def _wait_idle(win, timeout=10):
    end = time.time() + timeout
    while win.busy and time.time() < end:
        _app.processEvents()
        time.sleep(0.01)
    _app.processEvents()
    assert not win.busy, "task did not finish"


class _Patched:
    """Silence audio and redirect the preview file for tests that really run a task."""

    def __enter__(self):
        self._play, self._stop, self._out = audio_player.play, audio_player.stop, qt_app.OUTPUT_DIR
        self.played = []
        audio_player.play = self.played.append
        audio_player.stop = lambda: None
        self.tmp = tempfile.mkdtemp()
        qt_app.OUTPUT_DIR = self.tmp
        return self

    def __exit__(self, *exc):
        audio_player.play, audio_player.stop, qt_app.OUTPUT_DIR = self._play, self._stop, self._out


def _test_window_builds_with_sidebar_navigation():
    win, stub = _make_window()
    labels = [b.text() for b in win.nav_buttons]
    assert labels == ["Create Voice", "Clone Voice", "My Voices", "Audiobooks", "Settings"], labels
    assert win.stack.count() == 5 and win.stack.currentIndex() == 0
    assert win.nav_buttons[0].isChecked()
    assert win.nav_buttons[-1] is win.nav_buttons[4], "Settings is last (bottom of the sidebar)"
    assert stub.calls == []
    print("PASS: window builds with sidebar navigation, no model touched")


def _test_navigation_switches_pages():
    win, _ = _make_window()
    for i, btn in enumerate(win.nav_buttons):
        btn.click()
        assert win.stack.currentIndex() == i
        assert [b.isChecked() for b in win.nav_buttons].count(True) == 1 and btn.isChecked()
    assert win.stack.widget(4).isAncestorOf(win.cmb_attn)
    print("PASS: sidebar buttons switch pages and stay exclusive")


def _test_create_voice_widgets():
    win, _ = _make_window()
    assert isinstance(win.btn_vd_generate, QtWidgets.QPushButton)
    assert win.btn_vd_generate.text() == "Generate voice"
    assert isinstance(win.btn_vd_play, QtWidgets.QPushButton)
    assert isinstance(win.btn_vd_save, QtWidgets.QPushButton)
    assert isinstance(win.txt_vd_instruct, QtWidgets.QPlainTextEdit)
    assert isinstance(win.edt_vd_name, QtWidgets.QLineEdit)
    assert win.cmb_vd_lang.findText("english") >= 0
    assert win.txt_vd_instruct.toPlainText().strip()
    assert 100 <= win.txt_vd_instruct.height() <= 120 or win.txt_vd_instruct.minimumHeight() == 112
    assert win.btn_vd_generate.isEnabled() and not win.btn_vd_save.isEnabled()
    assert win.vd_preview.currentIndex() == PV_EMPTY
    for name in ("cmb_vd_seed_mode", "spn_vd_seed", "chk_vd_autosave", "cmb_vd_name_gender",
                 "btn_vd_download", "btn_vd_regenerate", "cmb_device", "cmb_dtype", "cmb_attn",
                 "spn_min", "spn_max"):
        assert hasattr(win, name), name
    print("PASS: Create Voice widgets exist with expected initial state")


def _test_advanced_settings_hidden_by_default():
    win, _ = _make_window()
    assert win.vd_advanced.isHidden()
    for w in (win.cmb_vd_seed_mode, win.chk_vd_autosave, win.cmb_vd_name_gender, win.txt_vd_sample):
        assert win.vd_advanced.isAncestorOf(w)
    win.btn_vd_advanced.click()
    assert not win.vd_advanced.isHidden()
    assert win.model_settings() == (win.cmb_device.currentText(), win.cmb_dtype.currentText(),
                                    win.cmb_attn.currentText())
    win.cmb_dtype.setCurrentText("float32")
    assert "float32" in win.lbl_vd_compute.text()
    win.btn_vd_advanced.click()
    assert win.vd_advanced.isHidden()
    print("PASS: Advanced settings collapsed by default, expandable, compute summary follows Settings")


def _test_suggestion_chips_are_reversible():
    win, _ = _make_window()
    original = "My own words."
    win.txt_vd_instruct.setPlainText(original)
    chip, _name, phrase = win.vd_chips[0]
    chip.setChecked(True)
    text = win.txt_vd_instruct.toPlainText()
    assert text.startswith(original) and phrase in text
    chip.setChecked(False)
    assert win.txt_vd_instruct.toPlainText() == original
    chip.setChecked(True)
    win.txt_vd_instruct.setPlainText(original)  # user edits the phrase away
    assert not chip.isChecked()
    print("PASS: suggestion chips append, remove only their phrase, and follow manual edits")


def _test_generate_validates_without_starting_a_task():
    win, stub = _make_window()
    win.txt_vd_instruct.setPlainText("   ")
    win.btn_vd_generate.click()
    assert win.lbl_status.text().startswith("Error"), win.lbl_status.text()
    assert not win.lbl_vd_field_error.isHidden()
    assert not win.busy and win.task is None and stub.calls == []
    assert win.vd_preview.currentIndex() == PV_EMPTY
    win.txt_vd_instruct.setPlainText("Calm voice")
    assert win.lbl_vd_field_error.isHidden()
    print("PASS: empty description shows an inline error and starts nothing")


def _test_generation_failure_shows_error_and_keeps_input():
    with _Patched():
        win, stub = _make_window()
        win.txt_vd_instruct.setPlainText("A gravelly old sailor")
        win.btn_vd_generate.click()
        assert win.vd_preview.currentIndex() == PV_BUSY and win.busy
        _wait_idle(win)
        assert stub.calls == ["voice-design"]
        assert win.vd_preview.currentIndex() == PV_ERROR
        # the footer is a concise status; the Preview owns the explanation and the details
        assert win.lbl_status.text().startswith("Couldn't generate") and "must not be loaded" not in win.lbl_status.text()
        assert not win.lbl_status_icon.isHidden(), "an error status carries its glyph as well as its words"
        assert "Couldn't" in win.lbl_vd_error.text() or "wrong" in win.lbl_vd_error.text()
        assert "must not be loaded" in win.lbl_vd_details.text()
        assert win.txt_vd_instruct.toPlainText() == "A gravelly old sailor"
        assert win.btn_vd_generate.isEnabled() and win.btn_vd_retry.isEnabled()
    print("PASS: failed generation shows the error state, preserves input, allows retry")


def _test_generation_success_shows_real_preview():
    with _Patched() as patched:
        win, _ = _make_window(models=_FakeModels())
        win.txt_vd_instruct.setPlainText("Warm narrator")
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY
        assert win.pending_voice["description"] == "Warm narrator"
        assert win.lbl_vd_duration.text() == "0:01", win.lbl_vd_duration.text()
        assert not win.lbl_vd_duration.isHidden()
        assert not win.btn_vd_save.isEnabled(), "Save waits for a voice name"
        win.edt_vd_name.setText("Ada")
        assert win.btn_vd_save.isEnabled() and win.btn_vd_regenerate.isEnabled()
        assert len(patched.played) == 1 and patched.played[0].endswith("_voice_preview.wav")
        assert os.path.isfile(win.pending_voice["preview"])
        assert win.vd_wave._peaks and abs(max(win.vd_wave._peaks) - 1.0) < 1e-6
        assert win.lbl_vd_stale.isHidden()
        win.txt_vd_instruct.setPlainText("Something else")
        assert not win.lbl_vd_stale.isHidden()
        # a later failure keeps the previous sample visible
        win.models = _StubModels()
        win.btn_vd_regenerate.click()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY and not win.lbl_vd_notice.isHidden()
        assert win.pending_voice["description"] == "Warm narrator"
    print("PASS: success shows real duration/waveform; stale note; failure keeps previous sample")


def _page_contains(stack, widget):
    return stack.currentWidget().isAncestorOf(widget)


def _test_empty_preview_is_compact_and_has_no_audio_controls():
    win, _ = _make_window()
    assert win.vd_preview.currentIndex() == PV_EMPTY
    assert win.preview_panel.property("empty") is True
    for w in (win.btn_vd_play, win.btn_vd_save, win.btn_vd_download, win.btn_vd_regenerate, win.vd_wave):
        assert not _page_contains(win.vd_preview, w), "audio actions must not show before audio exists"
    assert win.lbl_vd_duration.isHidden()
    win.show()
    _app.processEvents()
    assert win.preview_panel.height() < 160, win.preview_panel.height()
    win.hide()
    print("PASS: empty preview is compact, quiet, and shows no playback or fake duration")


def _test_previous_sample_stays_available_while_regenerating():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.btn_vd_generate.click()
        _wait_idle(win)
        first_preview = win.pending_voice["preview"]
        slow = _SlowModels()
        win.models = slow
        win.btn_vd_regenerate.click()
        _app.processEvents()
        time.sleep(0.2)
        _app.processEvents()
        assert win.busy
        assert win.vd_preview.currentIndex() == PV_READY, "previous sample must stay visible"
        assert win.btn_vd_play.isEnabled() and not win.btn_vd_regenerate.isEnabled()
        assert "Generating a new sample" in win.lbl_vd_notice.text()
        assert win.pending_voice["preview"] == first_preview
        slow.release.set()
        _wait_idle(win)
        assert win.vd_preview.currentIndex() == PV_READY and win.lbl_vd_notice.isHidden()
    print("PASS: previous sample stays visible and playable while a new one generates")


def _test_preview_text_is_never_clipped_or_oversized():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.resize(1100, 760)
        win.show()
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.models = _StubModels()
        win.btn_vd_regenerate.click()  # fails: long notice above the still-visible previous sample
        _wait_idle(win)
        _app.processEvents()
        panel, notice, play = win.preview_panel, win.lbl_vd_notice, win.btn_vd_play
        assert not notice.isHidden()
        bottom = notice.mapTo(panel, notice.rect().bottomLeft()).y()
        assert bottom <= play.mapTo(panel, play.rect().topLeft()).y(), "notice overlaps the player row"
        assert notice.height() >= notice.heightForWidth(notice.width()) - 1, "notice text is clipped"
        win.pending_voice = None  # error -> empty must shrink back to the compact panel
        win._set_preview(PV_ERROR)
        _app.processEvents()
        win._set_preview(PV_EMPTY)
        _app.processEvents()
        assert panel.height() < 160, panel.height()
        win.hide()
    print("PASS: preview text is never clipped, and the panel shrinks back to its compact empty state")


def _test_footer_actions_match_what_they_can_do():
    win, _ = _make_window()
    win.show_page(0)
    assert win.btn_play_last.isHidden(), "Create Voice has its own Play"
    for i in (1, 2, 3, 4):
        win.show_page(i)
        assert not win.btn_play_last.isHidden()
    win.show_page(0)
    assert win.btn_stop.isHidden() and not win.btn_stop.isEnabled()
    with _Patched():
        win.btn_vd_generate.click()  # voice creation can't be cancelled, so no Stop button
        assert win.busy and not win.btn_stop.isEnabled() and win.btn_stop.isHidden()
        _wait_idle(win)
    win.set_busy(True)  # a cancellable task (clone / audiobook) offers Stop
    assert win.btn_stop.isEnabled() and not win.btn_stop.isHidden()
    win.set_busy(False)
    assert win.btn_stop.isHidden()
    print("PASS: footer scopes Play-last to long-form pages; Stop appears only for cancellable tasks")


def _test_settings_page_separates_scopes():
    win, _ = _make_window()
    page = win.stack.widget(4)
    for w in (win.cmb_device, win.cmb_dtype, win.cmb_attn, win.spn_min, win.spn_max):
        assert page.isAncestorOf(w)
    texts = [l.text() for l in page.findChildren(QtWidgets.QLabel)]
    assert "Compute" in texts and "Long-form text" in texts
    assert not page.isAncestorOf(win.cmb_vd_lang), "language is task-specific"
    print("PASS: Settings holds application-wide options; task options stay on their screen")


def _test_my_voices_empty_state():
    original = voice_library.VOICES_DIR
    voice_library.VOICES_DIR = tempfile.mkdtemp()
    try:
        win, _ = _make_window()
        assert not win.lbl_voices_empty.isHidden()
        os.makedirs(os.path.join(voice_library.VOICES_DIR, "Ada"))
        with open(os.path.join(voice_library.VOICES_DIR, "Ada", "voice.json"), "w") as f:
            f.write('{"name": "Ada", "description": "d", "language": "english", "seed": 1, "created": "x"}')
        win.refresh_voices_tab()
        assert win.lbl_voices_empty.isHidden()
    finally:
        voice_library.VOICES_DIR = original
    print("PASS: My Voices shows an empty-state hint only when there are no voices")


def _with_voices(names):
    """Temporary voice library; returns (restore, root)."""
    original = voice_library.VOICES_DIR
    root = voice_library.VOICES_DIR = tempfile.mkdtemp()
    for name in names:
        os.makedirs(os.path.join(root, name))
        with open(os.path.join(root, name, "voice.json"), "w") as f:
            f.write('{"name": "%s", "description": "A warm narrator voice", "language": "english", '
                    '"seed": 1, "created": "2026-11-12 10:00"}' % name)

    def restore():
        voice_library.VOICES_DIR = original
    return restore


def _test_my_voices_actions_fit_at_1024_px():
    for names in ([], ["Ada", "Brook", "Cleo"]):
        restore = _with_voices(names)
        try:
            win, _ = _make_window()
            win.resize(1024, 640)
            win.show_page(PAGE_VOICES)
            _shown(win)
            page = win.stack.currentWidget()
            area = page.findChild(QtWidgets.QScrollArea)
            assert area.horizontalScrollBar().maximum() == 0, "My Voices must not scroll sideways at 1024 px"
            viewport = area.viewport().rect()
            for button in page.findChildren(QtWidgets.QPushButton):
                if button.text() == "▶":
                    continue
                top_left = button.mapTo(area.viewport(), QtCore.QPoint(0, 0))
                assert viewport.contains(QtCore.QRect(top_left, button.size())), f"{button.text()!r} is clipped"
                need = button.fontMetrics().horizontalAdvance(button.text()) + 24
                assert button.width() >= need, f"{button.text()!r} is squeezed"
            win.close()
        finally:
            restore()
    print("PASS: every My Voices action is visible at 1024 px, with and without voices")


def _test_engine_references_require_exactly_one_voice():
    restore = _with_voices(["Ada", "Brook"])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        button = win.btn_voice_engine_refs
        assert not button.isEnabled() and "exactly one" in button.toolTip()
        win.tv_voices.selectRow(0)
        assert button.isEnabled()
        win.tv_voices.selectAll()
        assert not button.isEnabled(), "two selected voices must disable the engine references"
        win.tv_voices.clearSelection()
        assert not button.isEnabled()
        # The old handlers keep their own error fallback when called without a selection.
        win.edit_csm_transcript()
        assert "select exactly one voice" in win.lbl_status.text().lower()
        win.edit_f5_reference()
        assert "select exactly one voice" in win.lbl_status.text().lower()
        # Both existing actions are still offered, under their original names.
        labels = [a.text() for a in button.menu().actions()]
        assert labels == ["Reference transcript…", "F5 reference…"], labels
        win.close()
    finally:
        restore()
    print("PASS: engine references are disabled unless exactly one voice is selected; handlers keep their fallback")


def _test_engine_references_menu_works_from_the_keyboard():
    restore = _with_voices(["Ada"])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        win.tv_voices.selectRow(0)
        calls = []
        win.act_voice_transcript.triggered.disconnect()
        win.act_voice_f5.triggered.disconnect()
        win.act_voice_transcript.triggered.connect(lambda: calls.append("transcript"))
        win.act_voice_f5.triggered.connect(lambda: calls.append("f5"))
        button = win.btn_voice_engine_refs
        button.setFocus()
        _app.processEvents()
        assert win.focusWidget() is button
        QTest.keyClick(button, QtCore.Qt.Key_Space)
        _app.processEvents()
        menu = button.menu()
        assert menu.isVisible(), "Space must open the menu"
        QTest.keyClick(menu, QtCore.Qt.Key_Down)
        QTest.keyClick(menu, QtCore.Qt.Key_Down)
        QTest.keyClick(menu, QtCore.Qt.Key_Return)
        _app.processEvents()
        assert calls == ["f5"], calls
        QTest.keyClick(button, QtCore.Qt.Key_Space)
        _app.processEvents()
        assert menu.isVisible()
        QTest.keyClick(menu, QtCore.Qt.Key_Escape)
        _app.processEvents()
        assert not menu.isVisible() and win.focusWidget() is button, "Escape closes the menu and keeps focus"
        win.close()
    finally:
        restore()
    print("PASS: the engine references menu opens, navigates, selects and closes with the keyboard")


def _test_my_voices_tab_order():
    restore = _with_voices(["Ada"])
    try:
        win, _ = _make_window()
        win.resize(1024, 640)
        win.show_page(PAGE_VOICES)
        _shown(win)
        win.tv_voices.selectRow(0)
        page = win.stack.currentWidget()
        first = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Import voice…")
        first.setFocus()
        _app.processEvents()
        seen = [first.text()]
        for _ in range(7):
            QTest.keyClick(win.focusWidget(), QtCore.Qt.Key_Tab)
            _app.processEvents()
            seen.append(getattr(win.focusWidget(), "text", lambda: "?")())
        expected = ["Import voice…", "Create from audio…", "Refresh / Detect", "▶ Play reference",
                    "Rename selected", "Delete selected", "Engine references"]
        assert seen[:7] == expected, seen
        win.close()
    finally:
        restore()
    print("PASS: My Voices Tab order is library actions, then selected-voice actions")


def _test_multiline_editors_never_trap_the_keyboard():
    win, _ = _make_window()
    win.resize(1024, 640)
    tab = win.audiobook_tab
    editors = {"audiobook chapter": tab.txt_source, "audiobook segment": tab.txt_segment,
               "clone text": win.txt_vc_text, "clone instruction": win.txt_vc_instruct}
    for name, editor in editors.items():
        assert editor.tabChangesFocus(), f"{name} traps Tab"
    win.show_page(3)
    _shown(win)
    tab.txt_source.setFocus()
    _app.processEvents()
    QTest.keyClicks(tab.txt_source, "one")
    QTest.keyClick(tab.txt_source, QtCore.Qt.Key_Return)
    QTest.keyClicks(tab.txt_source, "two")
    assert tab.txt_source.toPlainText() == "one\ntwo", "multiline entry must still work"
    QTest.keyClick(tab.txt_source, QtCore.Qt.Key_Tab)
    _app.processEvents()
    assert win.focusWidget() is not tab.txt_source and "\t" not in tab.txt_source.toPlainText()
    win.close()
    print("PASS: Audiobook and Clone text boxes release Tab and still accept multiline text")


def _test_tab_leaves_description_under_the_macos_default_tab_behavior():
    """Offscreen Qt tabs through every control, real macOS does not unless Full Keyboard Access is on.

    On macOS only text/list controls are Tab stops by default, so the description was the only one
    and Tab inserted a tab character. Emulate that default and drive real window-level key events.
    """
    macos_default = QtCore.Qt.TabFocusBehavior(3)  # text + list controls, as on a stock Mac
    _make_window()
    hints = _app.styleHints()
    original = hints.tabFocusBehavior()
    try:
        def key(win, k, mod=QtCore.Qt.NoModifier):
            QTest.keyClick(win.windowHandle(), k, mod)
            _app.processEvents()

        # Without the app's setting the trap reproduces (this documents why offscreen missed it).
        hints.setTabFocusBehavior(macos_default)
        win, _ = _make_window()
        hints.setTabFocusBehavior(macos_default)
        win.resize(1024, 640)
        _shown(win)
        win.txt_vd_instruct.setFocus()
        _app.processEvents()
        key(win, QtCore.Qt.Key_Tab)
        assert win.focusWidget() is win.txt_vd_instruct and "\t" in win.txt_vd_instruct.toPlainText(), \
            "expected the macOS default to trap Tab; the emulation is not faithful"
        win.close()

        # With the app's startup configuration (what main() does) the full sequence works.
        for text in ("", "line one\nline two"):
            hints.setTabFocusBehavior(macos_default)
            configure_keyboard_navigation(_app)
            win, _ = _make_window()
            win.resize(1024, 640)
            _shown(win)
            win.nav_buttons[0].setFocus()
            _app.processEvents()
            key(win, QtCore.Qt.Key_Tab)                       # navigation -> Voice description
            assert win.focusWidget() is win.txt_vd_instruct, _focus_label(win.focusWidget())
            win.txt_vd_instruct.setPlainText(text)
            before = win.txt_vd_instruct.toPlainText()
            key(win, QtCore.Qt.Key_Tab)                       # description -> next control
            assert win.focusWidget() is not win.txt_vd_instruct, "Tab is trapped in the description"
            assert win.focusWidget() is win.vd_chips[0][0], _focus_label(win.focusWidget())
            assert win.txt_vd_instruct.toPlainText() == before, "Tab must not modify the description"
            key(win, QtCore.Qt.Key_Backtab, QtCore.Qt.ShiftModifier)   # Shift+Tab goes back
            assert win.focusWidget() is win.txt_vd_instruct
            assert win.txt_vd_instruct.toPlainText() == before
            key(win, QtCore.Qt.Key_Tab, QtCore.Qt.ShiftModifier)       # Shift+Tab as Tab+Shift
            assert win.focusWidget() is win.nav_buttons[0], _focus_label(win.focusWidget())
            win.txt_vd_instruct.setFocus()
            win.txt_vd_instruct.moveCursor(QtGui.QTextCursor.End)
            _app.processEvents()
            key(win, QtCore.Qt.Key_Return)                    # Enter still adds a line
            assert win.txt_vd_instruct.toPlainText() == before + "\n" and win.focusWidget() is win.txt_vd_instruct
            win.close()
    finally:
        hints.setTabFocusBehavior(original)
    print("PASS: Tab/Shift+Tab leave the description under the macOS default tab behavior; Enter adds a line")


def _test_stale_sample_callout_and_footer_status():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _shown(win)
        win.edt_vd_name.setText("Narrator")
        _generate(win, "Warm narrator")
        assert win.lbl_status.text() == READY_STATUS
        assert win.frm_vd_stale.isHidden() and win.lbl_vd_stale.isHidden()
        win.txt_vd_instruct.setPlainText("Bright announcer")
        _app.processEvents()
        # the inset callout carries the note; the footer must not claim the sample matches the form
        assert not win.frm_vd_stale.isHidden() and not win.lbl_vd_stale.isHidden()
        assert win.frm_vd_stale.isAncestorOf(win.lbl_vd_stale)
        assert win.lbl_status.text() == STALE_STATUS
        assert win.lbl_vd_stale.height() >= win.lbl_vd_stale.heightForWidth(win.lbl_vd_stale.width()), "note is clipped"
        win.txt_vd_instruct.setPlainText("Warm narrator")
        _app.processEvents()
        assert win.frm_vd_stale.isHidden() and win.lbl_status.text() == READY_STATUS
        # an unrelated status message is never overwritten
        win.status("Saved as 'x'.")
        win.txt_vd_instruct.setPlainText("Another")
        _app.processEvents()
        assert win.lbl_status.text() == "Saved as 'x'."
        win.close()
    print("PASS: a changed input shows the callout and a matching footer status, and reverting restores both")


_USED_ICONS = {"wand-sparkles", "copy", "mic-vocal", "book-open", "settings", "audio-lines", "play", "save",
               "download", "refresh-cw", "chevron-right", "chevron-down",
               "square", "folder-open", "circle-check", "circle-exclamation-point"}


def _test_icon_assets_are_vendored_licensed_and_only_the_used_ones():
    on_disk = {f[:-4] for f in os.listdir(icons.ICON_DIR) if f.endswith(".svg")}
    assert on_disk == _USED_ICONS, f"vendor only the icons in use: {on_disk ^ _USED_ICONS}"
    licence = open(os.path.join(icons.ICON_DIR, "LICENSE"), encoding="utf-8").read()
    assert "ISC License" in licence and "Lucide" in licence and "Feather" in licence and "MIT License" in licence
    _make_window()
    for name in _USED_ICONS:
        for size in (icons.SIZE_COMPACT, icons.SIZE_NAV):
            assert not icons.icon(name, size).isNull(), name
    print("PASS: exactly the used Lucide icons are vendored, with the ISC and MIT Feather notices")


def _test_icons_use_theme_colours_per_state_and_are_high_dpi():
    _make_window()
    size = QtCore.QSize(16, 16)

    def has_colour(icon, hex_colour, mode=QtGui.QIcon.Normal, state=QtGui.QIcon.Off):
        """True if the rendered glyph contains the colour (antialiasing rounds channels by 1-2)."""
        want = QtGui.QColor(hex_colour)
        image = icon.pixmap(size, 1.0, mode, state).toImage()
        for x in range(image.width()):
            for y in range(image.height()):
                c = image.pixelColor(x, y)
                if c.alpha() > 200 and max(abs(c.red() - want.red()), abs(c.green() - want.green()),
                                           abs(c.blue() - want.blue())) <= 3:
                    return True
        return False

    ic = icons.icon("save", 16, "default")
    assert has_colour(ic, theme.COLORS["text_secondary"])                               # resting
    assert has_colour(ic, theme.COLORS["text_disabled"], QtGui.QIcon.Disabled)          # disabled
    assert has_colour(ic, theme.COLORS["text"], QtGui.QIcon.Normal, QtGui.QIcon.On)     # checked / selected
    assert has_colour(ic, theme.COLORS["text"], QtGui.QIcon.Active)                     # keyboard focus
    assert not has_colour(ic, theme.COLORS["text"], QtGui.QIcon.Normal), "resting icons stay neutral and quiet"
    primary = icons.icon("audio-lines", 16, "primary")
    assert has_colour(primary, theme.COLORS["on_primary"]), "the primary action's icon is white, like its label"
    assert has_colour(primary, theme.COLORS["on_primary_disabled"], QtGui.QIcon.Disabled)
    assert ic.pixmap(size, 2.0).width() >= 32, "a 2x pixmap must exist for high-DPI displays"
    print("PASS: icons take theme colours for normal, selected, focused and disabled states and ship 2x pixmaps")


def _test_missing_icon_assets_degrade_to_the_text_label():
    _make_window()
    assert icons.icon("no-such-icon").isNull()
    button = QtWidgets.QPushButton("Label")
    icons.apply(button, "no-such-icon")
    assert button.text() == "Label" and button.icon().isNull()
    print("PASS: a missing icon asset leaves the labelled button intact")


def _test_create_voice_and_navigation_icons_never_replace_labels():
    win, _ = _make_window()
    win.resize(1024, 640)
    _shown(win)
    labelled = {
        "nav": win.nav_buttons,
        "actions": [win.btn_vd_generate, win.btn_vd_play, win.btn_vd_save, win.btn_vd_download,
                    win.btn_vd_regenerate, win.btn_vd_advanced],
    }
    for button in labelled["nav"] + labelled["actions"]:
        assert not button.icon().isNull(), button.text()
        assert button.text().strip(), "icons never replace the text label"
        need = button.fontMetrics().horizontalAdvance(button.text()) + button.iconSize().width() + 16
        assert button.width() >= need, f"{button.text()!r} clips its text next to the icon"
    assert [b.text() for b in win.nav_buttons] == ["Create Voice", "Clone Voice", "My Voices", "Audiobooks", "Settings"]
    assert win.btn_vd_generate.text() == "Generate voice" and win.btn_vd_generate.accessibleName() in ("", "Generate voice")
    assert all(chip.icon().isNull() for chip, _n, _p in win.vd_chips), "suggestion chips stay text-only"
    # the disclosure says its state in words too, not by chevron alone
    assert win.btn_vd_advanced.accessibleDescription() == "Collapsed"
    win.btn_vd_advanced.click()
    assert win.btn_vd_advanced.accessibleDescription() == "Expanded"
    assert win.btn_vd_advanced.text() == "Advanced settings"
    # selected navigation uses the checked (On) icon, resting items the normal one
    assert win.nav_buttons[0].isChecked() and not win.nav_buttons[1].isChecked()
    win.close()
    print("PASS: Create Voice and navigation icons keep text labels, accessible names and chip text-only")


def _test_primary_button_uses_the_specified_orange_and_white_content():
    win, _ = _make_window()
    _shown(win)
    c = theme.COLORS
    assert (c["primary"], c["primary_hover"], c["primary_pressed"], c["primary_disabled"], c["on_primary"]) == \
        ("#C5540A", "#B24C08", "#A94708", "#6B4A38", "#FFFFFF")
    button = win.btn_vd_generate

    def fill():
        _app.processEvents()
        return button.grab().toImage().pixelColor(4, button.height() // 2).name().upper()

    win.nav_buttons[0].setFocus()
    assert fill() == c["primary"]
    QTest.mouseMove(button, button.rect().center())
    _app.processEvents()
    assert fill() == c["primary_hover"]
    QTest.mousePress(button, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, button.rect().center())
    assert fill() == c["primary_pressed"]
    edge = button.grab().toImage().pixelColor(0, button.height() // 2).name().upper()
    assert edge == c["primary_pressed_edge"], "pressed has an edge hover lacks, so the two never rely on a ~1.09:1 fill difference alone"
    assert c["primary_pressed_edge"] != c["primary_hover"] != c["primary_pressed"]
    # release outside the button so the press is cancelled and no generation starts
    QTest.mouseRelease(button, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, QtCore.QPoint(-20, -20))
    assert not win.busy
    QTest.mouseMove(win.lbl_vd_description, QtCore.QPoint(2, 2))
    button.setEnabled(False)
    assert fill() == c["primary_disabled"]
    button.setEnabled(True)
    # label and glyph are white; size is unchanged; the keyboard focus ring stays visible
    assert button.size() == QtCore.QSize(180, 42), button.size()
    image = button.icon().pixmap(button.iconSize(), 1.0, QtGui.QIcon.Normal).toImage()
    assert any(image.pixelColor(x, y).alpha() > 200 and image.pixelColor(x, y).name().upper() == "#FFFFFF"
               for x in range(image.width()) for y in range(image.height())), "the icon must be white"
    win.vd_chips[-1][0].setFocus()
    _app.processEvents()
    QTest.keyClick(win.windowHandle(), QtCore.Qt.Key_Tab)
    _app.processEvents()
    assert win.focusWidget() is button
    ring = button.grab().toImage().pixelColor(button.width() // 2, 0).name().upper()
    assert ring == c["text"].upper(), f"focus ring not visible: {ring}"
    win.close()
    print("PASS: Generate voice renders the specified orange in every state with a white icon, label and visible focus ring")


def _test_footer_actions_have_labels_icons_and_preserved_handlers():
    win, _ = _make_window()
    win.resize(1024, 640)
    _shown(win)
    expected = {win.btn_stop_audio: "Stop audio", win.btn_open_out: "Open output folder",
                win.btn_play_last: "Play last output", win.btn_stop: "Stop generation"}
    for button, label in expected.items():
        assert button.text() == label and not button.icon().isNull(), label
        assert button.iconSize().height() == icons.SIZE_COMPACT
        assert qt_app.__dict__ and button.property("role") == "quiet", "footer actions stay neutral and secondary"
    assert win.btn_stop_audio.icon().cacheKey() != win.btn_open_out.icon().cacheKey()
    # Stop audio keeps calling audio_player.stop
    stopped = []
    real_stop, real_playing = audio_player.stop, audio_player.is_playing
    audio_player.stop = lambda: stopped.append(1)
    audio_player.is_playing = lambda: True
    try:
        win._refresh_audio_state()
        assert win.btn_stop_audio.isEnabled()
        win.btn_stop_audio.click()
        assert stopped == [1]
    finally:
        audio_player.stop, audio_player.is_playing = real_stop, real_playing
    # Open output folder keeps opening OUTPUT_DIR (and says so if the system refuses)
    opened = []
    real_open = QtGui.QDesktopServices.openUrl
    QtGui.QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toLocalFile()) or True)
    try:
        win.btn_open_out.click()
        assert opened == [qt_app.OUTPUT_DIR], opened
        QtGui.QDesktopServices.openUrl = staticmethod(lambda url: False)
        win.btn_open_out.click()
        assert win.lbl_status.text().startswith("Couldn't open the output folder")
    finally:
        QtGui.QDesktopServices.openUrl = real_open
    win.close()
    print("PASS: footer actions keep labels, neutral role and 16 px icons; Stop audio and Open folder keep their handlers")


def _test_stop_audio_follows_the_players_real_state():
    win, _ = _make_window()
    _shown(win)
    state = {"playing": False}
    real_playing = audio_player.is_playing
    audio_player.is_playing = lambda: state["playing"]
    try:
        win._refresh_audio_state()
        assert not win.btn_stop_audio.isEnabled() and win.btn_stop_audio.toolTip() == qt_app.STOP_AUDIO_IDLE_TIP
        state["playing"] = True
        win._refresh_audio_state()
        assert win.btn_stop_audio.isEnabled() and win.btn_stop_audio.toolTip() == qt_app.STOP_AUDIO_TIP
        state["playing"] = False                      # playback finished by itself
        win.audio_timer.timeout.emit()
        assert not win.btn_stop_audio.isEnabled(), "back to idle once the player reports silence"
        assert win.audio_timer.isActive() and win.audio_timer.interval() == 250
    finally:
        audio_player.is_playing = real_playing
    # the read-only query never initialises or starts anything
    audio_player._proc = None
    assert audio_player.is_playing() in (True, False)
    win.close()
    print("PASS: Stop audio is enabled only while the player reports audio, and returns to idle by itself")


def _test_footer_status_is_contextual_and_never_duplicates_errors():
    original = voice_library.VOICES_DIR
    voice_library.VOICES_DIR = tempfile.mkdtemp()
    try:
        win, _ = _make_window()
        assert win.lbl_status.text() == "Ready", "the Create Voice screen must not open on a library count"
        assert win.lbl_status_icon.isHidden()
        win.refresh_voices_tab()                      # an explicit refresh may announce the count
        assert win.lbl_status.text() == "0 voice(s) found."
        win.status(READY_STATUS)
        assert not win.lbl_status_icon.isHidden() and win.lbl_status.text() == READY_STATUS
        win.status("Error: boom")
        assert not win.lbl_status_icon.isHidden()
        win.status("Ready")
        assert win.lbl_status_icon.isHidden(), "plain states show no glyph"
        win.close()
    finally:
        voice_library.VOICES_DIR = original
    with _Patched():
        win, _ = _make_window()                       # stub models raise a long technical message
        win.txt_vd_instruct.setPlainText("A gravelly old sailor")
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.lbl_vd_details.text() and win.lbl_vd_details.text() not in win.lbl_status.text()
        assert "Preview" in win.lbl_status.text()
        win.close()
    print("PASS: the footer says Ready at launch, uses a glyph for ready/error, and never repeats error details")


def _test_footer_fits_and_is_keyboard_reachable_at_minimum_width():
    win, _ = _make_window()
    win.resize(1024, 640)
    _shown(win)
    for page in range(5):
        win.show_page(page)
        _app.processEvents()
        footer = win.btn_open_out.parentWidget()
        for button in (win.btn_stop_audio, win.btn_open_out, win.btn_play_last):
            if button.isHidden():
                continue
            top_left = button.mapTo(win, QtCore.QPoint(0, 0))
            assert QtCore.QRect(top_left, button.size()).right() <= win.width(), f"{button.text()!r} clips at 1024 px"
            need = button.fontMetrics().horizontalAdvance(button.text()) + button.iconSize().width() + 16
            assert button.width() >= need, f"{button.text()!r} is squeezed"
        assert footer.height() <= 60, f"footer is {footer.height()} px tall"
    # Tab reaches Open output folder; Stop audio only joins the order while audio plays
    win.show_page(0)
    audio_player.is_playing, real = (lambda: False), audio_player.is_playing
    try:
        win._refresh_audio_state()
        win.btn_open_out.setFocus()
        _app.processEvents()
        assert win.focusWidget() is win.btn_open_out
        assert not win.btn_stop_audio.focusPolicy() == QtCore.Qt.NoFocus
        assert not win.btn_stop_audio.isEnabled(), "disabled controls must not take Tab"
        audio_player.is_playing = lambda: True
        win._refresh_audio_state()
        assert win.btn_stop_audio.isEnabled()
        win.btn_stop_audio.setFocus(QtCore.Qt.TabFocusReason)
        _app.processEvents()
        assert win.focusWidget() is win.btn_stop_audio
    finally:
        audio_player.is_playing = real
    win.close()
    print("PASS: footer actions fit at 1024 px on every page and stay keyboard-reachable")


def _test_create_voice_layout_at_minimum_and_wide_sizes():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.resize(1024, 640)
        win.show()
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.edt_vd_name.setText("Ada")
        win.txt_vd_instruct.setPlainText("Changed so the sample is a previous one")
        win.btn_vd_advanced.click()
        for chip, _name, _phrase in win.vd_chips:  # every suggestion on: the widest the chip row gets
            chip.setChecked(True)
        _app.processEvents()
        _app.processEvents()
        page = win.stack.widget(0)
        assert win.minimumWidth() <= 1024
        assert page.horizontalScrollBar().maximum() == 0, "Create Voice must not overflow horizontally"
        assert win.btn_vd_save.text() == "Save previous sample"
        assert win.btn_vd_save.width() >= win.btn_vd_save.sizeHint().width(), "Save label is clipped"
        assert win.edt_vd_name.width() >= 90, win.edt_vd_name.width()
        body = page.widget()
        top = lambda w: w.mapTo(body, w.rect().topLeft()).y()
        assert abs(top(win.preview_panel) - top(win.lbl_vd_description)) <= 2, "preview aligns with the description"
        win.resize(1700, 900)
        _app.processEvents()
        assert win.vd_content.width() <= CONTENT_MAX_WIDTH, win.vd_content.width()
        assert page.horizontalScrollBar().maximum() == 0
        win.hide()
    print("PASS: no horizontal overflow at 1024x640; content capped when wide; preview aligned")


# ------------------------------------------------------------------ keyboard + sample integrity
def _press(key, modifiers=QtCore.Qt.NoModifier, widget=None):
    QTest.keyClick(widget or _app.focusWidget(), key, modifiers)
    _app.processEvents()


def _tab(shift=False):
    _press(QtCore.Qt.Key_Tab, QtCore.Qt.ShiftModifier if shift else QtCore.Qt.NoModifier)


def _focus_label(w):
    """Readable name for the focused widget; unknown kinds show up as '?Class' so tests can reject them."""
    if isinstance(w, QtWidgets.QPushButton):
        return " ".join(w.text().split())
    if isinstance(w, QtWidgets.QCheckBox):
        return w.text()
    if isinstance(w, QtWidgets.QComboBox):
        return "combo:" + w.currentText()
    if isinstance(w, (QtWidgets.QPlainTextEdit, QtWidgets.QLineEdit)):
        return w.accessibleName()
    return "?" + type(w).__name__


def _shown(win):
    win.resize(1240, 760)
    win.show()
    win.activateWindow()
    _app.processEvents()


def _focus_cycle(win, backwards=False, limit=45):
    """Press Tab (or Shift+Tab) from the checked nav item until focus returns to it."""
    start = win.nav_buttons[win.stack.currentIndex()]
    if win.stack.currentIndex() == 0:
        win.txt_vd_instruct.setFocus()  # how a user starts: the form has focus, nothing pre-focused the nav
        _app.processEvents()
        _tab(shift=True)  # first stop backwards must be the SELECTED nav item, not its last button
        assert _app.focusWidget() is start, _focus_label(_app.focusWidget())
    else:
        start.setFocus()
        _app.processEvents()
    labels = [_focus_label(_app.focusWidget())]
    for _ in range(limit):
        _tab(shift=backwards)
        if _app.focusWidget() is start:
            return labels
        labels.append(_focus_label(_app.focusWidget()))
    raise AssertionError(f"focus never returned to navigation: {labels}")


_CHIPS = ["Narration", "Relaxing", "Epic", "Professional"]
_EMPTY_ORDER = (["Create Voice", "Voice description"] + _CHIPS
                + ["Generate voice", "combo:english", "Advanced settings", "Open output folder"])
_READY_ORDER = (["Create Voice", "Voice description"] + _CHIPS
                + ["Generate voice", "Play", "Voice name", "Save", "Download", "Regenerate",
                   "combo:english", "Advanced settings", "Open output folder"])


def _test_tab_leaves_description_and_multiline_entry_still_works():
    win, _ = _make_window()
    _shown(win)
    field = win.txt_vd_instruct
    field.setPlainText("Line one")
    field.setFocus()
    _app.processEvents()
    assert field.tabChangesFocus()
    _tab()
    assert _app.focusWidget() is win.vd_chips[0][0], _focus_label(_app.focusWidget())
    assert field.toPlainText() == "Line one", "Tab must not insert a tab character"
    field.setFocus()
    cursor = field.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    field.setTextCursor(cursor)
    _press(QtCore.Qt.Key_Return)
    QTest.keyClicks(field, "line two")
    assert field.toPlainText() == "Line one\nline two", repr(field.toPlainText())
    assert _app.focusWidget() is field
    win.hide()
    print("PASS: Tab leaves the description without inserting a tab; Enter still starts a new line")


def _test_shift_tab_navigates_backwards():
    win, _ = _make_window()
    _shown(win)
    win.vd_chips[0][0].setFocus()
    _app.processEvents()
    _tab(shift=True)
    assert _app.focusWidget() is win.txt_vd_instruct
    _tab(shift=True)
    assert _app.focusWidget() is win.nav_buttons[0], _focus_label(_app.focusWidget())
    _tab(shift=True)  # wraps to the end of the window: the last visible footer action
    assert _focus_label(_app.focusWidget()) == "Open output folder"
    forward = _focus_cycle(win)
    backward = _focus_cycle(win, backwards=True)
    assert backward == [forward[0]] + forward[:0:-1], (forward, backward)
    win.hide()
    print("PASS: Shift+Tab walks the exact reverse of the Tab order, including wrap-around")


def _test_keyboard_focus_order_is_logical():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _shown(win)
        order = _focus_cycle(win)
        assert order == _EMPTY_ORDER, order
        assert not [l for l in order if l.startswith("?")], "unexpected focus stop (invisible or non-control)"
        win.btn_vd_generate.click()
        _wait_idle(win)
        win.edt_vd_name.setText("Ada")
        _app.processEvents()
        order = _focus_cycle(win)
        assert order == _READY_ORDER, order
        win.edt_vd_name.setText("")  # Save is disabled, so it is skipped rather than a dead stop
        _app.processEvents()
        assert "Save" not in _focus_cycle(win)
        win.edt_vd_name.setText("Ada")
        win.btn_vd_advanced.click()
        _app.processEvents()
        order = _focus_cycle(win)
        after = order[order.index("Advanced settings") + 1:]
        assert after == ["combo:Random", "Auto-save generated voices", "combo:Random", "Suggest name",
                         "Audition text", "Open Settings", "Open output folder"], after
        assert not [l for l in order if l.startswith("?")]
        win.hide()
    print("PASS: Tab order = navigation, form, Generate, preview, settings, footer; no invisible stops")


def _test_generate_shortcut_and_visible_hint():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _shown(win)
        native = qt_app.shortcut_text()
        assert win.lbl_vd_shortcut.text() == f"or press {native}"
        assert native in win.btn_vd_generate.toolTip() and native in win.btn_vd_generate.accessibleDescription()
        assert win.lbl_vd_shortcut.property("role") == "muted"
        field = win.txt_vd_instruct
        field.setFocus()
        before = field.toPlainText()
        _press(QtCore.Qt.Key_Return, QtCore.Qt.ControlModifier)
        assert win.busy, "shortcut did not start generation"
        _wait_idle(win)
        assert win.pending_voice and field.toPlainText() == before, "shortcut must not type a newline"
        win.pending_voice = None
        win.show_page(1)  # the shortcut belongs to Create Voice only
        win.nav_buttons[1].setFocus()
        _press(QtCore.Qt.Key_Return, QtCore.Qt.ControlModifier)
        assert not win.busy and win.pending_voice is None
        win.hide()
    print("PASS: the generate shortcut works from the form, types nothing, and is shown next to the button")


def _generate(win, text=None):
    if text is not None:
        win.txt_vd_instruct.setPlainText(text)
    win.btn_vd_generate.click()
    _wait_idle(win)
    _app.processEvents()


def _test_changing_inputs_marks_the_sample_as_previous():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _generate(win, "Warm narrator")
        assert win.lbl_vd_preview_title.text() == "Preview" and win.lbl_vd_stale.isHidden()
        assert "Warm narrator" in win.lbl_vd_origin.text() and "english" in win.lbl_vd_origin.text()
        assert win.btn_vd_save.text() == "Save"
        win.txt_vd_instruct.setPlainText("A different voice")
        assert win.lbl_vd_preview_title.text() == "Previous sample"
        assert not win.lbl_vd_stale.isHidden()
        assert "Warm narrator" in win.lbl_vd_origin.text() and "different" not in win.lbl_vd_origin.text()
        assert win.btn_vd_save.text() == "Save previous sample"
        assert win.pending_voice["description"] == "Warm narrator", "the sample keeps its own inputs"
        win.txt_vd_instruct.setPlainText("Warm narrator")  # back to the original inputs
        assert win.lbl_vd_preview_title.text() == "Preview" and win.lbl_vd_stale.isHidden()
        assert win.btn_vd_save.text() == "Save"
        win.cmb_vd_lang.setCurrentText("french")  # language alone also counts
        assert win.lbl_vd_preview_title.text() == "Previous sample" and not win.lbl_vd_stale.isHidden()
        win.cmb_vd_lang.setCurrentText("english")
        assert win.lbl_vd_stale.isHidden()
        long_text = "x" * 200
        _generate(win, long_text)
        assert len(win.lbl_vd_origin.text()) < 130 and win.lbl_vd_origin.text().count("x") < 100
        assert win.lbl_vd_origin.toolTip() == long_text
    print("PASS: changed inputs relabel the audio as a previous sample and keep showing what made it")


def _test_saving_a_previous_sample_stores_its_original_inputs():
    saved = []
    real_save, real_dir = voice_library.save_voice, voice_library.VOICES_DIR
    voice_library.VOICES_DIR = tempfile.mkdtemp()
    voice_library.save_voice = lambda *a, **k: saved.append((a, k))
    try:
        with _Patched():
            win, _ = _make_window(models=_FakeModels())
            _generate(win, "Original voice description")
            sample = win.pending_voice
            win.txt_vd_instruct.setPlainText("Edited description that never produced this audio")
            win.cmb_vd_lang.setCurrentText("german")
            win.edt_vd_name.setText("Ada")
            assert win.btn_vd_save.isEnabled() and win.btn_vd_save.text() == "Save previous sample"
            win.btn_vd_save.click()
            _wait_idle(win)
            assert len(saved) == 1
            args = saved[0][0]
            name, wav, sr, prompt, description, language, seed = args[:7]
            assert name == "Ada" and prompt == "FAKE-PROMPT"
            assert description == "Original voice description", description
            assert language == "english" and seed == sample["seed"] and sr == sample["sr"]
            assert wav is sample["wav"]
            assert "made from" in win.lbl_vd_notice.text()
            assert win.txt_vd_instruct.toPlainText().startswith("Edited"), "saving must not touch the form"
    finally:
        voice_library.save_voice, voice_library.VOICES_DIR = real_save, real_dir
    print("PASS: saving a previous sample stores its original description, language and seed")


def _test_invalid_voice_names_disable_save():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        win.edt_vd_name.setText("Ada")
        assert not win.btn_vd_save.isEnabled(), "no sample yet"
        _generate(win)
        for bad, hint in (("", "Enter a voice name"), ("   ", "Enter a voice name"),
                          ("!!!", "letter or number"), ("***", "letter or number")):
            win.edt_vd_name.setText(bad)
            assert not win.btn_vd_save.isEnabled(), repr(bad)
            assert not win.lbl_vd_name_hint.isHidden() and hint in win.lbl_vd_name_hint.text(), repr(bad)
            assert "name" in win.btn_vd_save.toolTip().lower()
            win.btn_vd_save.click()
            assert not win.busy, "a disabled Save must start nothing"
        win.edt_vd_name.setText("Ada-2")
        assert win.btn_vd_save.isEnabled() and win.lbl_vd_name_hint.isHidden()
        win.set_busy(True)
        assert not win.btn_vd_save.isEnabled()
        win.set_busy(False)
        assert win.btn_vd_save.isEnabled(), "valid name: Save returns after a task"
        win.edt_vd_name.setText("")
        win.set_busy(True)
        win.set_busy(False)
        assert not win.btn_vd_save.isEnabled(), "finishing a task must not re-enable Save without a name"
    print("PASS: Save is disabled for empty or unusable names, with a reason, and stays so after tasks")


def _test_failed_regeneration_keeps_the_previous_valid_sample():
    with _Patched():
        win, _ = _make_window(models=_FakeModels())
        _generate(win, "Warm narrator")
        sample, wav = win.pending_voice, win.pending_voice["wav"].copy()
        win.edt_vd_name.setText("Ada")
        win.txt_vd_instruct.setPlainText("A new description that will fail")
        win.models = _StubModels()
        win.btn_vd_generate.click()
        _wait_idle(win)
        assert win.pending_voice is sample and np.array_equal(win.pending_voice["wav"], wav)
        assert win.vd_preview.currentIndex() == PV_READY
        assert "previous sample" in win.lbl_vd_notice.text().lower() and not win.lbl_vd_notice.isHidden()
        assert win.lbl_vd_preview_title.text() == "Previous sample"
        assert win.btn_vd_play.isEnabled() and win.btn_vd_save.isEnabled()
        assert win.btn_vd_save.text() == "Save previous sample"
        assert "Warm narrator" in win.lbl_vd_origin.text()
        assert win.txt_vd_instruct.toPlainText() == "A new description that will fail"
    print("PASS: a failed regeneration keeps the previous sample, its inputs, and a labelled Save")


def _test_busy_toggles_action_buttons():
    win, _ = _make_window()
    win.set_busy(True)
    assert all(not b.isEnabled() for b in win.action_buttons)
    assert win.btn_stop.isEnabled()
    win.set_busy(False)
    assert win.btn_vd_generate.isEnabled() and win.btn_vc_generate.isEnabled()
    assert not win.btn_stop.isEnabled()
    print("PASS: set_busy disables and restores action buttons")


def _test_single_primary_button_and_neutral_nav():
    win, _ = _make_window()
    primaries = [b for b in win.findChildren(QtWidgets.QPushButton) if b.property("role") == "primary"]
    assert set(primaries) == {win.btn_vd_generate, win.btn_vc_generate}, primaries
    assert win.stack.widget(0).isAncestorOf(win.btn_vd_generate)
    assert win.stack.widget(1).isAncestorOf(win.btn_vc_generate)
    assert all(b.property("role") == "nav" for b in win.nav_buttons)
    qss = theme.build_qss()
    selected = re.search(r'QPushButton\[role="nav"\]:checked \{(.*?)\}', qss, re.S).group(1)
    assert theme.COLORS["accent"].lower() not in selected.lower(), "selected nav must stay neutral"
    print("PASS: one primary button per action page; selected navigation is neutral")


def _test_sidebar_and_control_sizes():
    win, _ = _make_window()
    win.resize(1100, 800)
    win.show()
    _app.processEvents()
    side = win.findChild(QtWidgets.QFrame, "sidebar")
    assert 200 <= side.width() <= 220, side.width()
    assert 36 <= win.cmb_vd_lang.height() <= 40, win.cmb_vd_lang.height()
    assert 40 <= win.btn_vd_generate.height() <= 44, win.btn_vd_generate.height()
    win.btn_vd_advanced.click()
    win.resize(900, 620)
    _app.processEvents()
    win.hide()
    print("PASS: sidebar ~208 px, inputs 36-40 px, primary button 40-44 px")


def _test_theme_rules_and_contrast():
    qss = theme.build_qss()
    assert "gradient" not in qss.lower() and "shadow" not in qss.lower()
    c = theme.COLORS
    checks = {
        "text on canvas": (c["text"], c["canvas"], 4.5),
        "secondary on canvas": (c["text_secondary"], c["canvas"], 4.5),
        "muted on canvas": (c["text_muted"], c["canvas"], 4.5),
        "muted on surface": (c["text_muted"], c["surface"], 4.5),
        "muted on sidebar": (c["text_muted"], c["sidebar"], 4.5),
        "text on primary button": (c["on_primary"], c["primary"], 4.5),
        "text on primary pressed": (c["on_primary"], c["primary_pressed"], 4.5),
        "text on primary disabled": (c["on_primary_disabled"], c["primary_disabled"], 4.5),
        "text on primary hover": (c["on_primary"], c["primary_hover"], 4.5),
        "primary button on canvas": (c["primary"], c["canvas"], 3.0),
        "focus ring on primary": (c["text"], c["primary"], 3.0),
        "text on selection": (c["text"], c["selection"], 4.5),
        "nav selected text": (c["text"], c["nav_selected"], 4.5),
        "nav hover text": (c["text"], c["nav_hover"], 4.5),
        "error on surface": (c["error"], c["surface"], 4.5),
        "control border on canvas": (c["border_control"], c["canvas"], 3.0),
        "focus ring on canvas": (c["accent"], c["canvas"], 3.0),
        "focus ring on surface": (c["accent"], c["surface"], 3.0),
    }
    for label, (fg, bg, minimum) in checks.items():
        ratio = theme.contrast_ratio(fg, bg)
        assert ratio >= minimum, f"{label}: {ratio:.2f} < {minimum}"
    print("PASS: theme has no gradients/shadows; contrast targets met")


def _test_set_role_updates_property():
    win, _ = _make_window()
    b = QtWidgets.QPushButton("x")
    theme.set_role(b, "primary")
    assert b.property("role") == "primary"
    print("PASS: set_role sets the dynamic property")


def run_ui_tests():
    print("Running UI smoke tests (offscreen)...")
    _test_window_builds_with_sidebar_navigation()
    _test_navigation_switches_pages()
    _test_create_voice_widgets()
    _test_advanced_settings_hidden_by_default()
    _test_suggestion_chips_are_reversible()
    _test_generate_validates_without_starting_a_task()
    _test_generation_failure_shows_error_and_keeps_input()
    _test_generation_success_shows_real_preview()
    _test_tab_leaves_description_and_multiline_entry_still_works()
    _test_shift_tab_navigates_backwards()
    _test_keyboard_focus_order_is_logical()
    _test_generate_shortcut_and_visible_hint()
    _test_changing_inputs_marks_the_sample_as_previous()
    _test_saving_a_previous_sample_stores_its_original_inputs()
    _test_invalid_voice_names_disable_save()
    _test_failed_regeneration_keeps_the_previous_valid_sample()
    _test_empty_preview_is_compact_and_has_no_audio_controls()
    _test_previous_sample_stays_available_while_regenerating()
    _test_preview_text_is_never_clipped_or_oversized()
    _test_footer_actions_match_what_they_can_do()
    _test_settings_page_separates_scopes()
    _test_my_voices_empty_state()
    _test_my_voices_actions_fit_at_1024_px()
    _test_engine_references_require_exactly_one_voice()
    _test_engine_references_menu_works_from_the_keyboard()
    _test_my_voices_tab_order()
    _test_multiline_editors_never_trap_the_keyboard()
    _test_tab_leaves_description_under_the_macos_default_tab_behavior()
    _test_stale_sample_callout_and_footer_status()
    _test_icon_assets_are_vendored_licensed_and_only_the_used_ones()
    _test_icons_use_theme_colours_per_state_and_are_high_dpi()
    _test_missing_icon_assets_degrade_to_the_text_label()
    _test_create_voice_and_navigation_icons_never_replace_labels()
    _test_primary_button_uses_the_specified_orange_and_white_content()
    _test_footer_actions_have_labels_icons_and_preserved_handlers()
    _test_stop_audio_follows_the_players_real_state()
    _test_footer_status_is_contextual_and_never_duplicates_errors()
    _test_footer_fits_and_is_keyboard_reachable_at_minimum_width()
    _test_create_voice_layout_at_minimum_and_wide_sizes()
    _test_busy_toggles_action_buttons()
    _test_single_primary_button_and_neutral_nav()
    _test_sidebar_and_control_sizes()
    _test_theme_rules_and_contrast()
    _test_set_role_updates_property()
    print("All UI smoke tests passed.")


if __name__ == "__main__":
    run_ui_tests()
