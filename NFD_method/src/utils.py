"""
Shared dataclasses, drawing helpers, logger and timing utilities.

All pipeline results are represented as typed dataclasses so that
downstream code (webcam loop, enrollment) can access fields by name.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Core dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    """Result from FaceDetector for a single detected face."""
    bbox: np.ndarray                     # [x1, y1, x2, y2] float32
    score: float                         # detector confidence [0–1]
    landmarks: Optional[np.ndarray] = None  # shape (5, 2) float32, or None


@dataclass
class QualityResult:
    """Output from the quality gate module."""
    ok: bool
    score: float                         # 0.0–1.0, higher = better quality
    reasons: list[str] = field(default_factory=list)  # failure codes if ok=False


@dataclass
class LivenessResult:
    """Output from the liveness / anti-spoofing check."""
    is_live: bool
    score: float                         # 0 = spoof, 1 = live
    reason: str = "stub"


@dataclass
class SearchResult:
    """A single candidate returned by FAISS vector search."""
    employee_id: str
    name: str
    score: float                         # cosine similarity [0–1]
    image_path: str


@dataclass
class IdentificationResult:
    """Final identification decision for one detected face."""
    decision: str                        # ACCEPT | UNKNOWN | LOW_QUALITY | SPOOF_SUSPECTED
    employee_id: Optional[str]
    name: Optional[str]
    score: float                         # top-1 cosine similarity
    margin: float                        # top-1 score minus top-2 employee score
    quality: QualityResult
    liveness: LivenessResult
    latency_ms: float


# ── Colour palette (BGR) ─────────────────────────────────────────────────────

_COLORS: dict[str, tuple[int, int, int]] = {
    "ACCEPT":          (60, 220, 80),
    "UNKNOWN":         (30, 160, 255),
    "LOW_QUALITY":     (50,  50, 210),
    "SPOOF_SUSPECTED": (0,    0, 200),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _decision_color(decision: str) -> tuple[int, int, int]:
    return _COLORS.get(decision, _COLORS["UNKNOWN"])


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_face(
    frame: np.ndarray,
    det: FaceDetection,
    result: IdentificationResult,
) -> None:
    """
    Draw bounding box, identity label, score, and quality badge on *frame*
    in-place.  Colour-coded by decision outcome.
    """
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    color = _decision_color(result.decision)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Identity label above the box
    if result.decision == "ACCEPT":
        label = f"{result.name}  {result.score:.2f}"
    elif result.decision == "UNKNOWN":
        label = f"UNKNOWN  {result.score:.2f}"
    else:
        label = result.decision

    text_y = y1 - 10 if y1 > 30 else y2 + 22
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
    cv2.rectangle(frame, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, text_y), _FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Quality score small badge below box
    q_label = f"Q:{result.quality.score:.2f}"
    cv2.putText(frame, q_label, (x1, y2 + 16), _FONT, 0.45, color, 1, cv2.LINE_AA)

    # 5 facial landmarks dots
    if det.landmarks is not None:
        for lx, ly in det.landmarks.astype(int):
            cv2.circle(frame, (lx, ly), 2, color, -1)


def draw_hud(frame: np.ndarray, fps: float, latency_ms: float) -> None:
    """HUD overlay: FPS and per-frame latency in the top-left corner."""
    text = f"FPS: {fps:5.1f}   Latency: {latency_ms:.0f} ms"
    cv2.rectangle(frame, (0, 0), (280, 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 20), _FONT, 0.55, (180, 255, 180), 1, cv2.LINE_AA)


def draw_status(
    frame: np.ndarray,
    message: str,
    color: tuple[int, int, int] = (200, 200, 200),
) -> None:
    """Centered status message at the bottom of the frame."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(message, _FONT, 0.7, 2)
    x = (w - tw) // 2
    cv2.putText(frame, message, (x, h - 18), _FONT, 0.7, color, 2, cv2.LINE_AA)


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "nfd", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_identification(
    result: IdentificationResult,
    logger: logging.Logger,
) -> None:
    """Log one identification event to console."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        f"[IDENT] ts={ts} "
        f"emp={result.employee_id or 'UNKNOWN'} "
        f"name={result.name or '-'} "
        f"score={result.score:.3f} "
        f"margin={result.margin:.3f} "
        f"decision={result.decision} "
        f"quality={result.quality.score:.2f} "
        f"latency={result.latency_ms:.0f}ms"
    )


# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    """Context-manager wall-clock timer.

    Usage::

        with Timer() as t:
            do_something()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
