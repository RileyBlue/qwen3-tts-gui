"""Drama Studio Qt tab: cast management, LLM planning and ComfyUI rendering."""

import json
import os
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import audio_player
import icons
import minimax_drama as drama
import voice_library
from drama_studio import DramaStudio, voice_reference_path


class DramaTab(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.window = parent
        self.studio = DramaStudio()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_project_bar())
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self._build_cast()
        self._build_script()
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._sync_voice_preview_buttons)
        self.preview_timer.start()
        self._refresh_projects()
        self._on_project_changed()

    def _build_project_bar(self):
        row = QtWidgets.QHBoxLayout()
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.setMinimumWidth(220)
        self.project_combo.currentIndexChanged.connect(self._on_project_changed)
        btn_new = QtWidgets.QPushButton("New project")
        btn_new.clicked.connect(self._new_project)
        btn_open = QtWidgets.QPushButton("Open project folder")
        btn_open.clicked.connect(self._open_project)
        row.addWidget(QtWidgets.QLabel("Project"))
        row.addWidget(self.project_combo)
        row.addWidget(btn_new)
        row.addWidget(btn_open)
        row.addStretch(1)
        return row

    def _current_project(self):
        text = self.project_combo.currentText().strip() if hasattr(self, "project_combo") else ""
        return drama.slug(text or "my_drama")

    def _refresh_projects(self, select=None):
        projects = self.studio.list_projects()
        target = drama.slug(select) if select else (self.project_combo.currentText() or (projects[0] if projects else "my_drama"))
        if target not in projects:
            projects = sorted({*projects, target})
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        self.project_combo.addItems(projects)
        self.project_combo.setCurrentText(target)
        self.project_combo.blockSignals(False)

    def _new_project(self):
        name, accepted = QtWidgets.QInputDialog.getText(self, "New drama project", "Project name:")
        if not accepted or not name.strip():
            return
        slug_name = drama.slug(name)
        self.studio.references_dir(slug_name)
        self._refresh_projects(select=slug_name)
        self._on_project_changed()
        self.window.status(f"Selected drama project '{slug_name}'.")

    def _on_project_changed(self):
        self._new_character()
        self.refresh_cast()
        project = self._current_project()
        plan_path = drama.project_dir(project) / "plan.json"
        if plan_path.is_file():
            try:
                plan = self.studio.load_plan(project)
                self.plan_editor.setPlainText(json.dumps(plan, indent=2, ensure_ascii=False))
            except (OSError, ValueError):
                self.plan_editor.clear()
        else:
            self.plan_editor.clear()
        final = drama.project_dir(project) / f"{project}_final.mp4"
        self.output_path.setText(str(final) if final.is_file() else "")

    @staticmethod
    def _text(height=70):
        widget = QtWidgets.QPlainTextEdit()
        widget.setMaximumHeight(height)
        return widget

    def _build_cast(self):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)
        self.cast_table = QtWidgets.QTableWidget(0, 4)
        self.cast_table.setHorizontalHeaderLabels(("ID", "Name", "Voice", "Image"))
        self.cast_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cast_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.cast_table.horizontalHeader().setStretchLastSection(True)
        self.cast_table.itemSelectionChanged.connect(self._cast_selected)
        outer.addWidget(self.cast_table, 2)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form, 3)
        self.char_id = QtWidgets.QLineEdit()
        self.char_name = QtWidgets.QLineEdit()
        self.char_gender = QtWidgets.QComboBox()
        self.char_gender.addItems(("female", "male"))
        self.char_voice = QtWidgets.QComboBox()
        self.char_voice.currentIndexChanged.connect(self._sync_voice_preview_buttons)
        self.btn_voice_play = QtWidgets.QPushButton("Play")
        self.btn_voice_play.setToolTip("Preview the assigned saved voice reference audio")
        icons.apply(self.btn_voice_play, "play", icons.SIZE_COMPACT)
        self.btn_voice_play.clicked.connect(self._preview_voice)
        self.btn_voice_stop = QtWidgets.QPushButton("Stop")
        self.btn_voice_stop.setToolTip("Stop voice preview playback")
        icons.apply(self.btn_voice_stop, "square", icons.SIZE_COMPACT)
        self.btn_voice_stop.clicked.connect(self._stop_voice_preview)
        voice_row = QtWidgets.QHBoxLayout()
        voice_row.addWidget(self.char_voice, 1)
        voice_row.addWidget(self.btn_voice_play)
        voice_row.addWidget(self.btn_voice_stop)
        self.char_reference = QtWidgets.QLineEdit()
        self.char_appearance = self._text()
        self.char_tags = self._text(55)
        browse = QtWidgets.QPushButton("Choose reference image")
        browse.clicked.connect(self._browse_reference)
        self.char_preview = QtWidgets.QLabel("No reference image")
        self.char_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.char_preview.setMinimumHeight(150)
        form.addRow("ID", self.char_id)
        form.addRow("Name", self.char_name)
        form.addRow("Gender", self.char_gender)
        form.addRow("Assigned saved voice", voice_row)
        form.addRow("Reference image", self.char_reference)
        form.addRow(browse)
        form.addRow(self.char_preview)
        form.addRow("Appearance", self.char_appearance)
        form.addRow("Image generation tags", self.char_tags)
        buttons = QtWidgets.QHBoxLayout()
        for label, handler in (("New", self._new_character), ("Save character", self._save_character),
                               ("Delete", self._delete_character)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        form.addRow(buttons)
        self.tabs.addTab(page, "Cast")

    def refresh_voices(self):
        current = self.char_voice.currentData() or ""
        self.char_voice.clear()
        self.char_voice.addItem("(unassigned)", "")
        for name in voice_library.list_voices():
            self.char_voice.addItem(name, name)
        index = self.char_voice.findData(current)
        self.char_voice.setCurrentIndex(max(0, index))
        self._sync_voice_preview_buttons()

    def _sync_voice_preview_buttons(self):
        has_voice = bool(self.char_voice.currentData())
        self.btn_voice_play.setEnabled(has_voice)
        self.btn_voice_stop.setEnabled(audio_player.state() != "stopped")

    def _preview_voice(self):
        name = self.char_voice.currentData()
        if not name:
            self.window.status("Select an assigned saved voice to preview.")
            return
        path = voice_reference_path(name)
        if not path:
            self.window.status(f"Error: reference audio not found for '{name}'.")
            return
        err = self.window.play(path)
        if err:
            self.window.status(f"Error playing voice preview: {err}")
        else:
            self.window.status(f"Previewing saved voice '{name}'.")
        self._sync_voice_preview_buttons()

    def _stop_voice_preview(self):
        self.window.stop_audio()
        self._sync_voice_preview_buttons()

    def refresh_cast(self):
        self.refresh_voices()
        project = self._current_project()
        refs_dir = drama.project_references_dir(project)
        registry = self.studio.characters(project)
        self.cast_table.blockSignals(True)
        self.cast_table.setRowCount(0)
        for row, (cid, char) in enumerate(sorted(registry.items())):
            self.cast_table.insertRow(row)
            found = drama.find_reference(cid, char, refs_dir=refs_dir)
            values = (cid, char.get("name", cid), char.get("voice_name", ""),
                      char.get("reference") or ("generated" if found else "missing"))
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setData(QtCore.Qt.UserRole, cid)
                self.cast_table.setItem(row, col, item)
        self.cast_table.blockSignals(False)
        self.cast_table.resizeColumnsToContents()

    def _new_character(self):
        self.cast_table.clearSelection()
        for widget in (self.char_id, self.char_name, self.char_reference):
            widget.clear()
        self.char_appearance.clear()
        self.char_tags.clear()
        self.char_voice.setCurrentIndex(0)
        self.char_preview.setText("No reference image")

    def _cast_selected(self):
        row = self.cast_table.currentRow()
        item = self.cast_table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        project = self._current_project()
        char = self.studio.characters(project).get(item.data(QtCore.Qt.UserRole), {})
        self.char_id.setText(char.get("id", ""))
        self.char_name.setText(char.get("name", ""))
        self.char_gender.setCurrentText(char.get("gender", "female"))
        self.char_voice.setCurrentIndex(max(0, self.char_voice.findData(char.get("voice_name", ""))))
        self.char_reference.setText(char.get("reference", ""))
        self.char_appearance.setPlainText(char.get("appearance", ""))
        self.char_tags.setPlainText(char.get("ref_tags", ""))
        path = drama.find_reference(char.get("id", ""), char, refs_dir=drama.project_references_dir(project))
        pixmap = QtGui.QPixmap(str(path)) if path else QtGui.QPixmap()
        if pixmap.isNull():
            self.char_preview.setText("No reference image")
        else:
            self.char_preview.setPixmap(pixmap.scaled(180, 150, QtCore.Qt.KeepAspectRatio,
                                                      QtCore.Qt.SmoothTransformation))

    def _browse_reference(self):
        start_dir = str(self.studio.references_dir(self._current_project()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose character reference", start_dir,
                                                        "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.char_reference.setText(path)
            self.char_preview.setPixmap(QtGui.QPixmap(path).scaled(
                180, 150, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _save_character(self):
        try:
            item = self.studio.save_character(self._current_project(), {
                "id": self.char_id.text(), "name": self.char_name.text().strip(),
                "gender": self.char_gender.currentText(),
                "voice_name": self.char_voice.currentData(),
                "reference": self.char_reference.text().strip(),
                "appearance": self.char_appearance.toPlainText().strip(),
                "ref_tags": self.char_tags.toPlainText().strip()})
        except (OSError, ValueError) as exc:
            self.window.status(f"Could not save character: {exc}")
            return
        self.refresh_cast()
        self.window.status(f"Saved cast character {item['name']}.")

    def _delete_character(self):
        cid = self.char_id.text().strip()
        if cid and QtWidgets.QMessageBox.question(self, "Delete character?", f"Remove {cid} from the cast?") == QtWidgets.QMessageBox.Yes:
            self.studio.delete_character(self._current_project(), cid)
            self._new_character()
            self.refresh_cast()

    def _build_script(self):
        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        top = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton("Reload project plan")
        load.clicked.connect(self._load_project)
        top.addWidget(QtWidgets.QLabel("Prose or story prompt (quoted dialogue must be supplied verbatim)"))
        top.addStretch(1)
        top.addWidget(load)
        outer.addLayout(top)
        self.source = QtWidgets.QPlainTextEdit()
        self.source.setPlaceholderText("Paste prose with the exact dialogue you want characters to speak…")
        outer.addWidget(self.source, 2)
        outer.addWidget(QtWidgets.QLabel("Editable segment plan JSON"))
        self.plan_editor = QtWidgets.QPlainTextEdit()
        self.plan_editor.setPlaceholderText("Generate a script, or paste an existing segment plan here.")
        outer.addWidget(self.plan_editor, 3)
        options = QtWidgets.QHBoxLayout()
        self.video_seed = QtWidgets.QSpinBox()
        self.video_seed.setRange(0, 2**31 - 1)
        self.lightning = QtWidgets.QCheckBox("4-step Lightning")
        self.force_render = QtWidgets.QCheckBox("Re-render existing clips")
        options.addWidget(QtWidgets.QLabel("Base video seed"))
        options.addWidget(self.video_seed)
        options.addWidget(self.lightning)
        options.addWidget(self.force_render)
        options.addStretch(1)
        outer.addLayout(options)
        buttons = QtWidgets.QHBoxLayout()
        self.action_buttons = []
        for label, handler in (("Generate script with LLM", self._generate_script),
                               ("Save edited plan", self._save_plan),
                               ("Generate missing images", self._generate_images),
                               ("Render final video", self._render_video)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(handler)
            buttons.addWidget(button)
            self.action_buttons.append(button)
        outer.addLayout(buttons)
        self.output_path = QtWidgets.QLineEdit()
        self.output_path.setReadOnly(True)
        open_output = QtWidgets.QPushButton("Open project folder")
        open_output.clicked.connect(self._open_project)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.output_path)
        row.addWidget(open_output)
        outer.addLayout(row)
        self.tabs.addTab(page, "Script & Render")

    def _plan(self):
        try:
            data = json.loads(self.plan_editor.toPlainText())
        except ValueError as exc:
            raise ValueError(f"The plan is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("The plan must be a JSON object.")
        return data

    def _run(self, work, done, message):
        self.window.status(message)
        self.window.run_task(lambda task: work(), done, stoppable=False)

    def _generate_script(self):
        source, project = self.source.toPlainText(), self._current_project()
        settings, key = self.window.drama_connection_settings(), self.window.drama_api_key()
        self._run(lambda: self.studio.generate_plan(source, project, settings, key),
                  self._plan_ready, "Generating the segment plan with the configured LLM…")

    def _plan_ready(self, plan):
        self.plan_editor.setPlainText(json.dumps(plan, indent=2, ensure_ascii=False))
        self.refresh_cast()
        self.window.status(f"Script ready: {len(plan['segments'])} segments. Review the cast and plan before rendering.")

    def _validated_plan(self):
        return self.studio.save_plan(self._plan(), self.source.toPlainText(), self._current_project())

    def _save_plan(self):
        try:
            plan = self._validated_plan()
        except (OSError, ValueError, drama.PlanError) as exc:
            self.window.status(f"Plan error: {exc}")
            return
        self.plan_editor.setPlainText(json.dumps(plan, indent=2, ensure_ascii=False))
        self.refresh_cast()
        self.window.status("Segment plan validated and saved.")

    def _load_project(self):
        try:
            plan = self.studio.load_plan(self._current_project())
        except (OSError, ValueError) as exc:
            self.window.status(f"Could not load project: {exc}")
            return
        self.plan_editor.setPlainText(json.dumps(plan, indent=2, ensure_ascii=False))
        self.refresh_cast()
        self.window.status("Project plan loaded.")

    def _generate_images(self):
        try:
            plan = self._validated_plan()
        except (ValueError, drama.PlanError) as exc:
            self.window.status(f"Plan error: {exc}")
            return
        settings, project = self.window.drama_connection_settings(), self._current_project()
        self._run(lambda: self.studio.generate_images(plan, project, settings), self._images_ready,
                  "Submitting missing character images to ComfyUI…")

    def _images_ready(self, result):
        paths, generated = result
        self.refresh_cast()
        message = f"Generated references for {', '.join(generated)}." if generated else "All character references already exist."
        self.window.status(message)

    def _render_video(self):
        try:
            plan = self._validated_plan()
        except (ValueError, drama.PlanError) as exc:
            self.window.status(f"Plan error: {exc}")
            return
        settings = self.window.drama_connection_settings()
        project = self._current_project()
        seed, lightning, force = self.video_seed.value(), self.lightning.isChecked(), self.force_render.isChecked()
        self._run(lambda: self.studio.render_video(plan, project, settings, seed, lightning, force),
                  self._video_ready, "Generating images and video clips in ComfyUI; this may take a while…")

    def _video_ready(self, path):
        self.output_path.setText(path)
        self.refresh_cast()
        self.window.status(f"Final video saved to {path}")

    def _open_project(self):
        path = self.studio.project_dir(self._current_project())
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
