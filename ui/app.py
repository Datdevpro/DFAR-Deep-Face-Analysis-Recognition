"""Modern Flet UI for Plan1 workflow."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import flet as ft

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class Plan1FletApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._proc: subprocess.Popen | None = None

        self.device = ft.Dropdown(
            label="Device",
            value="cpu",
            width=140,
            options=[ft.dropdown.Option("cpu"), ft.dropdown.Option("cuda")],
        )
        self.copies = ft.TextField(label="Aug Copies", value="5", width=140)
        self.train_source = ft.Dropdown(
            label="Train Source",
            value="aug",
            width=160,
            options=[ft.dropdown.Option("raw"), ft.dropdown.Option("aug")],
        )
        self.artifact = ft.TextField(
            label="Artifact Path",
            value=str(ROOT / "artifacts" / "svm_plan1.joblib"),
            expand=True,
        )
        self.log = ft.TextField(
            label="Output Log",
            multiline=True,
            read_only=True,
            min_lines=18,
            max_lines=22,
            expand=True,
        )
        self._build()

    def _build(self) -> None:
        self.page.title = "Plan1 Face Attendance - Flet Launcher"
        self.page.padding = 16
        self.page.theme_mode = ft.ThemeMode.DARK

        settings = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
                    ft.Row([self.device, self.copies, self.train_source], wrap=True),
                    self.artifact,
                ],
                spacing=10,
            ),
            padding=12,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        )

        actions = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Actions", size=18, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        [
                            self._btn("Open Raw Data Folder", self.open_raw_folder),
                            self._btn("Download Models", self.download_models),
                            self._btn("Augment", self.augment),
                            self._btn("Train Model", self.train),
                        ],
                        wrap=True,
                    ),
                    ft.Row(
                        [
                            self._btn("Run Live Demo", self.demo),
                            self._btn("Test Image", self.test_image),
                            self._btn("Run API", self.api),
                            self._btn("Stop Running Task", self.stop_task, ft.Colors.ERROR_CONTAINER),
                        ],
                        wrap=True,
                    ),
                ],
                spacing=10,
            ),
            padding=12,
            border_radius=12,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        )

        self.page.add(settings, actions, self.log)
        self._append("Ready.\n")

    def _btn(self, text: str, on_click, bgcolor: str | None = None) -> ft.ElevatedButton:
        return ft.ElevatedButton(content=text, on_click=on_click, bgcolor=bgcolor)

    def _append(self, text: str) -> None:
        self.log.value = (self.log.value or "") + text
        self.page.update()

    def _run(self, cmd: list[str], env: dict[str, str] | None = None) -> None:
        if self._proc and self._proc.poll() is None:
            self.page.open(ft.SnackBar(ft.Text("Task already running. Stop it first.")))
            self.page.update()
            return

        self._append(f"\n$ {' '.join(cmd)}\n")
        base_env = os.environ.copy()
        if env:
            base_env.update(env)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=base_env,
        )
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _stream_output(self) -> None:
        if not self._proc or not self._proc.stdout:
            return
        for line in self._proc.stdout:
            self.page.run_thread(lambda ln=line: self._append(ln))
        code = self._proc.wait()
        self.page.run_thread(lambda: self._append(f"\n[done] exit_code={code}\n"))

    def open_raw_folder(self, _):
        path = ROOT / "data" / "raw"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def download_models(self, _):
        self._run([PYTHON, "scripts/download_models.py"])

    def augment(self, _):
        copies = (self.copies.value or "5").strip()
        self._run(
            [
                PYTHON,
                "scripts/augment_dataset.py",
                "--source",
                str(ROOT / "data" / "raw"),
                "--out",
                str(ROOT / "data" / "aug"),
                "--copies",
                copies,
            ]
        )

    def train(self, _):
        src = ROOT / "data" / (self.train_source.value or "aug")
        self._run(
            [
                PYTHON,
                "scripts/train_svm.py",
                "--data",
                str(src),
                "--artifact",
                (self.artifact.value or "").strip(),
                "--device",
                (self.device.value or "cpu").strip(),
            ]
        )

    def demo(self, _):
        self._run(
            [
                PYTHON,
                "scripts/demo_webcam.py",
                "--artifact",
                (self.artifact.value or "").strip(),
                "--device",
                (self.device.value or "cpu").strip(),
            ]
        )

    def test_image(self, _):
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        image = filedialog.askopenfilename(
            title="Pick test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        root.destroy()
        if not image:
            return
        self._run(
            [
                PYTHON,
                "scripts/test_image.py",
                "--image",
                image,
                "--artifact",
                (self.artifact.value or "").strip(),
                "--device",
                (self.device.value or "cpu").strip(),
            ]
        )

    def api(self, _):
        self._run(
            [PYTHON, "scripts/run_api.py"],
            env={
                "PLAN1_SVM_PATH": (self.artifact.value or "").strip(),
                "PLAN1_DEVICE": (self.device.value or "cpu").strip(),
            },
        )

    def stop_task(self, _):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._append("[stopping task...]\n")
        else:
            self._append("[no running task]\n")


def main(page: ft.Page) -> None:
    Plan1FletApp(page)

