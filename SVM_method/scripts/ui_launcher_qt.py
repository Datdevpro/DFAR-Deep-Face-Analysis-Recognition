"""Entrypoint for PySide6 cyberpunk launcher."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_qt.app import launch


if __name__ == "__main__":
    launch()

