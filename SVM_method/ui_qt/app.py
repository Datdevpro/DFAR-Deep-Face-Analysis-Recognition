"""PySide6 launcher — four-quadrant console layout (Vision-style) + Plan1 actions."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QAction, QPixmap, QTextCursor  # QAction used for menus
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui_qt.theme import ACCENT, ERROR, MUTED, STYLESHEET, SUCCESS

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class LauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.proc: QProcess | None = None
        self._stdout_parts: list[str] = []
        self._capture_output = False
        self.setWindowTitle("Plan1 Vision Processor")
        self.resize(1400, 900)
        self.setStyleSheet(STYLESHEET)
        self._build()

    def _build(self) -> None:
        self._create_menus()
        central = QWidget()
        self.setCentralWidget(central)
        main_row = QHBoxLayout(central)
        main_row.setContentsMargins(10, 8, 10, 10)
        main_row.setSpacing(10)

        # Stack must exist before nav rail: first nav button setChecked() emits toggled → _on_nav.
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_dashboard_page())
        for title in ("Projects", "Workflows", "History", "Data"):
            self.stack.addWidget(self._placeholder_page(title))

        main_row.addWidget(self._build_nav_rail())
        main_row.addWidget(self.stack, 1)

        self._set_status("idle")
        self._set_clock()
        self.append_log("Ready. Select a workflow and press Run Processing.")

    def _create_menus(self) -> None:
        bar = QMenuBar(self)

        def stub_menu(name: str) -> None:
            m = bar.addMenu(name)
            a = QAction("(placeholder)", self)
            a.setEnabled(False)
            m.addAction(a)

        stub_menu("File")
        stub_menu("Edit")
        stub_menu("Processing")
        stub_menu("Logs")
        stub_menu("Settings")

        help_m = bar.addMenu("Help")
        about = QAction("About Plan1 Vision Processor", self)
        about.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "About",
                "Plan1 face pipeline: train SVM, augment, test, demo, API.\n"
                "Menu entries are placeholders except this dialog.",
            )
        )
        help_m.addAction(about)

        self.setMenuBar(bar)

    def _build_nav_rail(self) -> QFrame:
        rail = QFrame()
        rail.setObjectName("navRail")
        rail.setFixedWidth(168)
        v = QVBoxLayout(rail)
        v.setContentsMargins(10, 14, 10, 14)
        v.setSpacing(6)
        brand = QLabel("Plan1")
        brand.setObjectName("title")
        sub = QLabel("Vision Console")
        sub.setObjectName("subtle")
        v.addWidget(brand)
        v.addWidget(sub)
        v.addSpacing(12)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for i, label in enumerate(["Dashboard", "Projects", "Workflows", "History", "Data"]):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setObjectName("nav")
            btn.toggled.connect(lambda on, idx=i: self._on_nav(idx) if on else None)
            self.nav_group.addButton(btn)
            v.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        v.addStretch(1)
        ver = QLabel("v1.0")
        ver.setObjectName("subtle")
        v.addWidget(ver)
        return rail

    def _on_nav(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        self.append_log(f"NAV → {['Dashboard', 'Projects', 'Workflows', 'History', 'Data'][index]}")

    def _placeholder_page(self, title: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        msg = QLabel(f"{title}\n\nThis section is a UI placeholder.")
        msg.setObjectName("subtle")
        msg.setAlignment(Qt.AlignCenter)
        lay.addWidget(msg)
        return w

    def _build_dashboard_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        header = QFrame()
        header.setObjectName("headerpanel")
        hr = QHBoxLayout(header)
        hr.setContentsMargins(14, 10, 14, 10)
        t = QLabel("Plan1 Vision Processor")
        t.setObjectName("title")
        st = QLabel("Face detection, embeddings, SVM training & runtime")
        st.setObjectName("subtle")
        st.setWordWrap(True)
        col = QVBoxLayout()
        col.addWidget(t)
        col.addWidget(st)
        hr.addLayout(col, 1)
        self.status_dot = QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_text = QLabel("Idle")
        self.status_text.setObjectName("subtle")
        self.clock_label = QLabel("")
        self.clock_label.setObjectName("subtle")
        hr.addWidget(self.status_dot, 0, Qt.AlignVCenter)
        hr.addWidget(self.status_text, 0, Qt.AlignVCenter)
        hr.addSpacing(12)
        hr.addWidget(self.clock_label, 0, Qt.AlignVCenter)
        outer.addWidget(header)

        tl = self._panel_processing_config()
        tr = self._panel_images_results()
        bl = self._panel_execution()
        br = self._panel_logs()

        top_split = QSplitter(Qt.Orientation.Horizontal)
        top_split.setChildrenCollapsible(False)
        top_split.setHandleWidth(5)
        top_split.addWidget(tl)
        top_split.addWidget(tr)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)

        bot_split = QSplitter(Qt.Orientation.Horizontal)
        bot_split.setChildrenCollapsible(False)
        bot_split.setHandleWidth(5)
        bot_split.addWidget(bl)
        bot_split.addWidget(br)
        bot_split.setStretchFactor(0, 1)
        bot_split.setStretchFactor(1, 1)

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.setChildrenCollapsible(False)
        main_split.setHandleWidth(5)
        main_split.addWidget(top_split)
        main_split.addWidget(bot_split)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 1)
        outer.addWidget(main_split, 1)
        return page

    def _wrap_card(self, inner: QWidget) -> QFrame:
        card = QFrame()
        card.setObjectName("panelCard")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card.setMinimumSize(0, 0)
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 12, 12, 12)
        v.addWidget(inner)
        return card

    def _panel_processing_config(self) -> QFrame:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)
        title = QLabel("Processing configuration")
        title.setObjectName("sectionTitle")
        v.addWidget(title)

        gb = QGroupBox("Parameters")
        g = QVBoxLayout(gb)
        g.setSpacing(8)

        # ── Model selection ────────────────────────────────────────────────
        self.artifact = QLineEdit(str(ROOT / "artifacts" / "svm_plan1.joblib"))
        self.artifact.setReadOnly(True)
        self.artifact.setPlaceholderText("No model selected…")
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        btn_model = QPushButton("Select model…")
        btn_model.setObjectName("primary")
        btn_model.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        btn_model.clicked.connect(self._pick_artifact)
        btn_model.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        model_row.addWidget(btn_model)
        model_row.addWidget(self.artifact, 1)
        g.addLayout(model_row)

        # ── Runtime settings ───────────────────────────────────────────────
        self.output_mode = QComboBox()
        self.output_mode.addItems(["JSON + Visual", "JSON only", "Log only"])
        self._add_labeled(g, "Output format", self.output_mode)

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.train_src = QComboBox()
        self.train_src.addItems(["raw", "aug"])
        self.copies = QLineEdit("5")
        self._add_labeled(g, "Device", self.device)
        self._add_labeled(g, "Train data source", self.train_src)
        self._add_labeled(g, "Augment copies", self.copies)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_row.addWidget(QPushButton("Reset", clicked=self._reset_config_ui))
        g.addLayout(reset_row)

        v.addWidget(gb)
        v.addStretch(1)
        return self._wrap_card(w)

    def _panel_images_results(self) -> QFrame:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)

        title = QLabel("Images & results")
        title.setObjectName("sectionTitle")
        v.addWidget(title)

        # ── Compact image strip ────────────────────────────────────────────
        strip_row = QHBoxLayout()
        strip_row.setSpacing(6)
        self.image_list = QListWidget()
        self.image_list.setFixedHeight(56)
        self.image_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.image_list.setFlow(QListView.Flow.LeftToRight)
        self.image_list.setWrapping(False)
        self.image_list.setResizeMode(QListWidget.Adjust)
        self.image_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_list.setSpacing(4)
        self.image_list.currentItemChanged.connect(self._on_image_selected)
        strip_row.addWidget(self.image_list, 1)
        strip_btn = QVBoxLayout()
        strip_btn.setSpacing(3)
        btn_add = QPushButton("Add")
        btn_add.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        btn_add.clicked.connect(self.add_images)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self.clear_images)
        strip_btn.addWidget(btn_add)
        strip_btn.addWidget(btn_clr)
        strip_row.addLayout(strip_btn)
        v.addLayout(strip_row)

        # ── Preview split (stretches with the card) ────────────────────────
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(5)
        self.preview_in = QLabel("Input preview")
        self.preview_in.setObjectName("subtle")
        self.preview_in.setAlignment(Qt.AlignCenter)
        self.preview_in.setMinimumSize(0, 0)
        self.preview_in.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_in.setStyleSheet(f"border:1px solid {MUTED}; border-radius:8px; background:#0B0E14;")
        self.preview_in.setScaledContents(True)
        self.preview_out = QLabel("Result preview\n(run Test Image)")
        self.preview_out.setObjectName("subtle")
        self.preview_out.setAlignment(Qt.AlignCenter)
        self.preview_out.setMinimumSize(0, 0)
        self.preview_out.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_out.setStyleSheet(f"border:1px solid {MUTED}; border-radius:8px; background:#0B0E14;")
        self.preview_out.setWordWrap(True)
        split.addWidget(self._preview_frame("Selected input", self.preview_in))
        split.addWidget(self._preview_frame("Result summary", self.preview_out))
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        v.addWidget(split, 1)
        return self._wrap_card(w)

    def _preview_frame(self, caption: str, inner: QWidget) -> QWidget:
        box = QWidget()
        l = QVBoxLayout(box)
        l.setSpacing(4)
        cap = QLabel(caption)
        cap.setObjectName("subtle")
        l.addWidget(cap)
        l.addWidget(inner, 1)
        return box

    def _panel_execution(self) -> QFrame:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)
        title = QLabel("Execution control")
        title.setObjectName("sectionTitle")
        v.addWidget(title)

        self.workflow = QComboBox()
        self.workflow.addItems(
            [
                "Train SVM",
                "Augment dataset",
                "Test image (selected or pick)",
                "Live demo (webcam)",
                "Run API server",
                "Download models",
            ]
        )
        self._add_labeled(v, "Workflow", self.workflow)

        row1 = QHBoxLayout()
        self.btn_run = QPushButton("  Run processing")
        self.btn_run.setObjectName("run")
        self.btn_run.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_run.clicked.connect(self.run_processing)
        row1.addWidget(self.btn_run, 2)

        self.btn_stop = QPushButton("  Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_BrowserStop))
        self.btn_stop.clicked.connect(self.stop_task)
        row1.addWidget(self.btn_stop, 1)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        b_save = QPushButton("Save results")
        b_save.setObjectName("save")
        b_save.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        b_save.clicked.connect(self._placeholder_save_results)
        row2.addWidget(b_save, 1)
        b_batch = QPushButton("Batch folder")
        b_batch.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        b_batch.clicked.connect(self._placeholder_batch)
        row2.addWidget(b_batch, 1)
        b_queue = QPushButton("Configure queue")
        b_queue.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        b_queue.clicked.connect(self._placeholder_queue)
        row2.addWidget(b_queue, 1)
        v.addLayout(row2)

        row3 = QHBoxLayout()
        btn_raw = QPushButton("Open raw data")
        btn_raw.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        btn_raw.clicked.connect(self.open_raw)
        row3.addWidget(btn_raw)
        row3.addStretch(1)
        v.addLayout(row3)

        stat_fr = QFrame()
        stat_fr.setObjectName("hudcard")
        sr = QVBoxLayout(stat_fr)
        sr.setContentsMargins(10, 10, 10, 10)
        self.status_line = QLabel("Status: Idle")
        self.status_line.setObjectName("subtle")
        sr.addWidget(self.status_line)
        self.current_task = QLabel("No task")
        self.current_task.setObjectName("sectionTitle")
        sr.addWidget(self.current_task)
        self.exit_code = QLabel("Exit: -")
        self.exit_code.setObjectName("subtle")
        sr.addWidget(self.exit_code)
        self.busy = QProgressBar()
        self.busy.setRange(0, 100)
        self.busy.setValue(0)
        self.busy.setTextVisible(True)
        sr.addWidget(self.busy)
        v.addWidget(stat_fr)
        v.addStretch(1)
        return self._wrap_card(w)

    def _panel_logs(self) -> QFrame:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(8)
        title = QLabel("Process logs & output")
        title.setObjectName("sectionTitle")
        v.addWidget(title)
        self.log = QTextEdit()
        self.log.setObjectName("logPanel")
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Timestamped log output…")
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.log, 1)
        return self._wrap_card(w)

    def _add_labeled(self, layout: QVBoxLayout, text: str, widget: QWidget) -> None:
        row = QVBoxLayout()
        row.setSpacing(4)
        lb = QLabel(text)
        lb.setObjectName("subtle")
        row.addWidget(lb)
        row.addWidget(widget)
        layout.addLayout(row)

    def _pick_artifact(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model artifact",
            str(ROOT / "artifacts"),
            "Joblib files (*.joblib);;All files (*)"
        )
        if path:
            self.artifact.setText(path)
            self.append_log(f"Model selected: {path}")

    def _reset_config_ui(self) -> None:
        self.artifact.setText(str(ROOT / "artifacts" / "svm_plan1.joblib"))
        self.device.setCurrentIndex(0)
        self.train_src.setCurrentIndex(0)
        self.copies.setText("5")
        self.output_mode.setCurrentIndex(0)
        self.append_log("CONFIG reset (UI defaults).")


    def _placeholder_save_results(self) -> None:
        self.append_log("PLACEHOLDER: Save results (wire export path when needed).")

    def _placeholder_batch(self) -> None:
        self.append_log("PLACEHOLDER: Batch process folder.")

    def _placeholder_queue(self) -> None:
        self.append_log("PLACEHOLDER: Configure queue.")

    def append_log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text.rstrip()}\n"
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(line)
        self.log.moveCursor(QTextCursor.End)

    def _set_status(self, status: str) -> None:
        if status == "running":
            self.status_text.setText("Running")
            self.status_dot.setStyleSheet(f"background:{ACCENT}; border-radius:5px;")
        elif status == "success":
            self.status_text.setText("Success")
            self.status_dot.setStyleSheet(f"background:{SUCCESS}; border-radius:5px;")
        elif status == "error":
            self.status_text.setText("Error")
            self.status_dot.setStyleSheet(f"background:{ERROR}; border-radius:5px;")
        else:
            self.status_text.setText("Idle")
            self.status_dot.setStyleSheet(f"background:{MUTED}; border-radius:5px;")

    def _set_clock(self) -> None:
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M"))

    def set_running(self, running: bool) -> None:
        if running:
            self._set_status("running")
            self.busy.setRange(0, 0)
            self.status_line.setText("Status: Running…")
        else:
            self._set_status("idle")
            self.busy.setRange(0, 100)
            self.busy.setValue(0)
            self.status_line.setText("Status: Idle")
        self._set_clock()

    def add_images(self, *_args) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add images", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        for f in files:
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.UserRole, f)
            self.image_list.addItem(item)
        if files:
            self.image_list.setCurrentRow(self.image_list.count() - 1)
            self.append_log(f"Added {len(files)} image(s) to strip.")

    def clear_images(self, *_args) -> None:
        self.image_list.clear()
        self.preview_in.clear()
        self.preview_in.setText("Input preview")
        self.preview_out.setText("Output / result preview")
        self.append_log("Cleared image strip.")

    def _on_image_selected(self, cur: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if cur is None:
            return
        path = cur.data(Qt.UserRole)
        if not path:
            return
        pix = QPixmap(str(path))
        if not pix.isNull():
            self.preview_in.setPixmap(pix)
        else:
            self.preview_in.setText("Could not load image")

    def run_processing(self, *_args) -> None:
        idx = self.workflow.currentIndex()
        if idx == 0:
            self.train_model()
        elif idx == 1:
            self.augment()
        elif idx == 2:
            self.test_image()
        elif idx == 3:
            self.live_demo()
        elif idx == 4:
            self.run_api()
        else:
            self.download_models()

    def _run(self, cmd: list[str], env: dict[str, str] | None = None, *, capture_for_preview: bool = False) -> None:
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Task Running", "Stop current task first.")
            return
        self._stdout_parts = []
        self._capture_output = capture_for_preview
        self.current_task.setText(Path(cmd[1]).name if len(cmd) > 1 else cmd[0])
        self.append_log(f"RUN $ {' '.join(cmd)}")
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        qenv = self.proc.processEnvironment()
        for k, v in os.environ.items():
            qenv.insert(k, v)
        if env:
            for k, v in env.items():
                qenv.insert(k, v)
        self.proc.setProcessEnvironment(qenv)
        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.readyReadStandardError.connect(self._on_stderr)
        self.proc.finished.connect(self._on_finished)
        self.set_running(True)
        self.proc.start(cmd[0], cmd[1:])

    def _on_stdout(self) -> None:
        if self.proc:
            chunk = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
            self.append_log(chunk.rstrip())
            if getattr(self, "_capture_output", False):
                self._stdout_parts.append(chunk)

    def _on_stderr(self) -> None:
        if self.proc:
            chunk = bytes(self.proc.readAllStandardError()).decode(errors="ignore")
            self.append_log(chunk.rstrip())
            if getattr(self, "_capture_output", False):
                self._stdout_parts.append(chunk)

    def _on_finished(self, code: int, _status) -> None:
        self.set_running(False)
        self.exit_code.setText(f"Exit: {code}")
        self.current_task.setText("No task")
        if getattr(self, "_capture_output", False):
            blob = "".join(self._stdout_parts).strip()
            if blob:
                self.preview_out.setText(blob[-4000:] if len(blob) > 4000 else blob)
            self._capture_output = False
        if code == 0:
            self._set_status("success")
            self.append_log(f"OK [done] exit_code={code}")
        else:
            self._set_status("error")
            self.append_log(f"ERR [done] exit_code={code}")

    def open_raw(self, *_args) -> None:
        path = ROOT / "data" / "raw"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def download_models(self, *_args) -> None:
        self._run([PYTHON, "scripts/download_models.py"])

    def augment(self, *_args) -> None:
        self._run(
            [
                PYTHON,
                "scripts/augment_dataset.py",
                "--source",
                str(ROOT / "data" / "raw"),
                "--out",
                str(ROOT / "data" / "aug"),
                "--copies",
                self.copies.text().strip() or "5",
            ]
        )

    def train_model(self, *_args) -> None:
        src = ROOT / "data" / self.train_src.currentText().strip()
        self._run(
            [
                PYTHON,
                "scripts/train_svm.py",
                "--data",
                str(src),
                "--artifact",
                self.artifact.text().strip(),
                "--device",
                self.device.currentText().strip(),
            ]
        )

    def live_demo(self, *_args) -> None:
        self._run(
            [
                PYTHON,
                "scripts/demo_webcam.py",
                "--artifact",
                self.artifact.text().strip(),
                "--device",
                self.device.currentText().strip(),
            ]
        )

    def test_image(self, *_args) -> None:
        image: str | None = None
        item = self.image_list.currentItem()
        if item is not None:
            image = item.data(Qt.UserRole)
        if not image:
            image, _ = QFileDialog.getOpenFileName(self, "Pick test image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not image:
            self.append_log("Test image cancelled (no file).")
            return
        self._run(
            [
                PYTHON,
                "scripts/test_image.py",
                "--image",
                image,
                "--artifact",
                self.artifact.text().strip(),
                "--device",
                self.device.currentText().strip(),
            ],
            capture_for_preview=True,
        )

    def run_api(self, *_args) -> None:
        self._run(
            [PYTHON, "scripts/run_api.py"],
            env={
                "PLAN1_SVM_PATH": self.artifact.text().strip(),
                "PLAN1_DEVICE": self.device.currentText().strip(),
            },
        )

    def stop_task(self, *_args) -> None:
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self._set_status("error")
            self.append_log("ERR [stopping task…]")
        else:
            self.append_log("INFO [no running task]")


def launch() -> None:
    app = QApplication(sys.argv)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec())
