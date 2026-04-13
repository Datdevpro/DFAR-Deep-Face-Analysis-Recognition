"""PySide6 launcher with modern structured layout and existing Plan1 functions."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
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
        self.setWindowTitle("Plan1 Face Attendance")
        self.resize(1320, 820)
        self.setStyleSheet(STYLESHEET)
        self._build()

    def _build(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setSpacing(8)
        outer.setContentsMargins(12, 12, 12, 12)
        self.setCentralWidget(root)

        header = QFrame()
        header.setObjectName("headerpanel")
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(14, 10, 14, 10)
        title_col = QVBoxLayout()
        title = QLabel("Plan1 Face Attendance")
        title.setObjectName("title")
        subtitle = QLabel("Training, testing, live demo and API control")
        subtitle.setObjectName("subtle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col, 1)
        self.status_dot = QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_text = QLabel("Idle")
        self.status_text.setObjectName("subtle")
        self.clock_label = QLabel("")
        self.clock_label.setObjectName("subtle")
        header_row.addWidget(self.status_dot, 0, Qt.AlignVCenter)
        header_row.addWidget(self.status_text, 0, Qt.AlignVCenter)
        header_row.addSpacing(16)
        header_row.addWidget(self.clock_label, 0, Qt.AlignVCenter)
        outer.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(8)
        side.addWidget(self._section_title("Main Actions"))
        side.addWidget(self._btn("Train Model", self.train_model, role="primary", icon=QStyle.SP_MediaPlay))
        side.addWidget(self._btn("Augment Dataset", self.augment, role="secondary", icon=QStyle.SP_BrowserReload))
        side.addWidget(self._btn("Test Image", self.test_image, role="secondary", icon=QStyle.SP_FileDialogContentsView))
        side.addWidget(self._btn("Run Live Demo", self.live_demo, role="secondary", icon=QStyle.SP_ComputerIcon))
        side.addWidget(self._btn("Run API", self.run_api, role="secondary", icon=QStyle.SP_DriveNetIcon))
        side.addWidget(self._btn("Download Models", self.download_models, role="secondary", icon=QStyle.SP_DialogSaveButton))
        side.addSpacing(8)
        side.addWidget(self._btn("Open Raw Data Folder", self.open_raw, role="ghost", icon=QStyle.SP_DirOpenIcon))
        side.addStretch(1)
        side.addWidget(self._btn("Stop Task", self.stop_task, role="danger", icon=QStyle.SP_BrowserStop))

        mainpanel = QFrame()
        mainpanel.setObjectName("mainpanel")
        main = QVBoxLayout(mainpanel)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)
        hud = QFrame()
        hud.setObjectName("hudcard")
        hud_row = QHBoxLayout(hud)
        hud_row.setContentsMargins(10, 10, 10, 10)
        self.current_task = QLabel("No task")
        self.current_task.setObjectName("sectionTitle")
        self.exit_code = QLabel("Exit: -")
        self.exit_code.setObjectName("subtle")
        self.busy = QProgressBar()
        self.busy.setRange(0, 100)
        self.busy.setValue(0)
        self.busy.setTextVisible(False)
        hud_row.addWidget(self.current_task, 1)
        hud_row.addWidget(self.exit_code)
        hud_row.addWidget(self.busy, 1)
        main.addWidget(hud)
        self.log = QTextEdit()
        self.log.setObjectName("logPanel")
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Terminal output stream...")
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main.addWidget(self.log, 1)

        settingspanel = QFrame()
        settingspanel.setObjectName("settingspanel")
        settings = QVBoxLayout(settingspanel)
        settings.setContentsMargins(12, 12, 12, 12)
        settings.setSpacing(10)
        settings.addWidget(self._section_title("Configuration"))

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.train_src = QComboBox()
        self.train_src.addItems(["raw", "aug"])
        self.copies = QLineEdit("5")
        self.artifact = QLineEdit(str(ROOT / "artifacts" / "svm_plan1.joblib"))
        self._add_labeled_field(settings, "Device", self.device)
        self._add_labeled_field(settings, "Train Source", self.train_src)
        self._add_labeled_field(settings, "Augment Copies", self.copies)
        self._add_labeled_field(settings, "Artifact Path", self.artifact)
        settings.addStretch(1)

        splitter.addWidget(sidebar)
        splitter.addWidget(mainpanel)
        splitter.addWidget(settingspanel)
        splitter.setSizes([260, 780, 280])
        outer.addWidget(splitter, 1)

        self._set_status("idle")
        self._set_clock()
        self.append_log("Ready.")

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionTitle")
        return label

    def _add_labeled_field(self, parent_layout: QVBoxLayout, label: str, widget: QWidget) -> None:
        row = QVBoxLayout()
        row.setSpacing(4)
        lb = QLabel(label)
        lb.setObjectName("subtle")
        row.addWidget(lb)
        row.addWidget(widget)
        parent_layout.addLayout(row)

    def _btn(self, text: str, handler, role: str = "secondary", icon: QStyle.StandardPixmap | None = None) -> QPushButton:
        btn = QPushButton(text)
        if icon is not None:
            btn.setIcon(self.style().standardIcon(icon))
        if role in {"primary", "ghost", "danger"}:
            btn.setObjectName(role)
        btn.clicked.connect(handler)
        return btn

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
        else:
            self._set_status("idle")
            self.busy.setRange(0, 100)
            self.busy.setValue(0)
        self._set_clock()

    def _run(self, cmd: list[str], env: dict[str, str] | None = None) -> None:
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Task Running", "Stop current task first.")
            return
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
            self.append_log(bytes(self.proc.readAllStandardOutput()).decode(errors="ignore"))

    def _on_stderr(self) -> None:
        if self.proc:
            self.append_log(bytes(self.proc.readAllStandardError()).decode(errors="ignore"))

    def _on_finished(self, code: int, _status) -> None:
        self.set_running(False)
        self.exit_code.setText(f"Exit: {code}")
        self.current_task.setText("No task")
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
        image, _ = QFileDialog.getOpenFileName(self, "Pick test image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if not image:
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
            ]
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
            self.append_log("ERR [stopping task...]")
        else:
            self.append_log("INFO [no running task]")


def launch() -> None:
    app = QApplication(sys.argv)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec())
