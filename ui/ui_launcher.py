"""Simple desktop launcher for Plan1 workflow."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class Plan1UI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Plan1 Face Attendance Launcher")
        self.geometry("980x650")
        self._proc: subprocess.Popen | None = None
        self._build_widgets()

    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        self.device_var = tk.StringVar(value="cpu")
        self.copies_var = tk.StringVar(value="5")
        self.train_src_var = tk.StringVar(value="aug")
        self.artifact_var = tk.StringVar(value=str(ROOT / "artifacts" / "svm_plan1.joblib"))

        ttk.Label(top, text="Device").grid(row=0, column=0, sticky="w")
        ttk.Combobox(top, textvariable=self.device_var, values=["cpu", "cuda"], width=8).grid(
            row=0, column=1, padx=5, sticky="w"
        )
        ttk.Label(top, text="Aug Copies").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.copies_var, width=6).grid(row=0, column=3, padx=5, sticky="w")
        ttk.Label(top, text="Train Source").grid(row=0, column=4, sticky="w")
        ttk.Combobox(top, textvariable=self.train_src_var, values=["raw", "aug"], width=8).grid(
            row=0, column=5, padx=5, sticky="w"
        )

        ttk.Label(top, text="Artifact").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.artifact_var, width=80).grid(
            row=1, column=1, columnspan=5, padx=5, pady=(8, 0), sticky="we"
        )

        buttons = ttk.Frame(self, padding=(10, 0))
        buttons.pack(fill=tk.X)

        ttk.Button(buttons, text="Open Raw Data Folder", command=self.open_raw_folder).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Download Models", command=self.download_models).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Augment", command=self.augment).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Train Model", command=self.train).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Run Live Demo", command=self.demo).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Test Image", command=self.test_image).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Run API", command=self.api).pack(side=tk.LEFT, padx=4, pady=6)
        ttk.Button(buttons, text="Stop Running Task", command=self.stop_task).pack(side=tk.LEFT, padx=4, pady=6)

        self.log = scrolledtext.ScrolledText(self, wrap=tk.WORD, font=("Consolas", 10))
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._append("Ready.\n")

    def _append(self, text: str) -> None:
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _run(self, cmd: list[str], env: dict[str, str] | None = None) -> None:
        if self._proc and self._proc.poll() is None:
            messagebox.showwarning("Task Running", "Stop current task first.")
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
            self.after(0, self._append, line)
        code = self._proc.wait()
        self.after(0, self._append, f"\n[done] exit_code={code}\n")

    def open_raw_folder(self) -> None:
        path = ROOT / "data" / "raw"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))

    def download_models(self) -> None:
        self._run([PYTHON, "scripts/download_models.py"])

    def augment(self) -> None:
        copies = self.copies_var.get().strip() or "5"
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

    def train(self) -> None:
        src = ROOT / "data" / self.train_src_var.get().strip()
        self._run(
            [
                PYTHON,
                "scripts/train_svm.py",
                "--data",
                str(src),
                "--artifact",
                self.artifact_var.get().strip(),
                "--device",
                self.device_var.get().strip(),
            ]
        )

    def demo(self) -> None:
        self._run(
            [
                PYTHON,
                "scripts/demo_webcam.py",
                "--artifact",
                self.artifact_var.get().strip(),
                "--device",
                self.device_var.get().strip(),
            ]
        )

    def test_image(self) -> None:
        image = filedialog.askopenfilename(
            title="Pick test image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not image:
            return
        self._run(
            [
                PYTHON,
                "scripts/test_image.py",
                "--image",
                image,
                "--artifact",
                self.artifact_var.get().strip(),
                "--device",
                self.device_var.get().strip(),
            ]
        )

    def api(self) -> None:
        self._run(
            [PYTHON, "scripts/run_api.py"],
            env={"PLAN1_SVM_PATH": self.artifact_var.get().strip(), "PLAN1_DEVICE": self.device_var.get().strip()},
        )

    def stop_task(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._append("[stopping task...]\n")
        else:
            self._append("[no running task]\n")


if __name__ == "__main__":
    app = Plan1UI()
    app.mainloop()
