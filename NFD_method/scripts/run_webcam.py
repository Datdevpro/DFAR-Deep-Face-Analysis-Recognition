"""
Launch the real-time webcam face identification demo.

Usage
-----
python scripts/run_webcam.py [--camera 0]

Controls inside the window:
  q  — quit
  s  — save current frame as debug/frame_<timestamp>.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.utils import setup_logger
from src.webcam_demo import WebcamDemo

logger = setup_logger("run_webcam")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run real-time face attendance webcam demo.")
    ap.add_argument(
        "--camera", type=int, default=CONFIG.webcam_index,
        help="OpenCV camera index (default: 0).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG.webcam_index = args.camera

    # Validate index exists before starting
    if not CONFIG.data.faiss_index_path.is_file():
        logger.error(
            f"FAISS index not found: {CONFIG.data.faiss_index_path}\n"
            "Run enroll_employee.py first to enroll at least one employee."
        )
        sys.exit(1)

    logger.info(f"Starting webcam demo (camera={args.camera}) …")
    demo = WebcamDemo.from_config(CONFIG)
    demo.run()


if __name__ == "__main__":
    main()
