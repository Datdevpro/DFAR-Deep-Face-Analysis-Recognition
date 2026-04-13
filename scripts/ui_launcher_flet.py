"""Entrypoint for modern Flet launcher UI."""

from __future__ import annotations

import sys
from pathlib import Path

import flet as ft

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.app import main


if __name__ == "__main__":
    ft.app(target=main)

