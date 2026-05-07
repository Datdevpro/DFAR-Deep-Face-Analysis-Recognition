"""Stage 2 — landmark-based synthetic accessories (dlib 68 points)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import dlib


def _mean_point(shape, indices: list[int]) -> tuple[float, float]:
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in indices])
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def draw_glasses_overlay(bgr: np.ndarray, shape, color=(40, 40, 40), thickness: int = 2) -> None:
    """Simple 'glasses' frame from eye landmarks (in-place)."""
    le = _mean_point(shape, list(range(36, 42)))
    re = _mean_point(shape, list(range(42, 48)))
    dx = re[0] - le[0]
    dy = re[1] - le[1]
    if abs(dx) < 1e-3:
        return
    dist = float(np.hypot(dx, dy))
    eye_r = int(max(4, dist * 0.35))
    c1 = (int(le[0]), int(le[1]))
    c2 = (int(re[0]), int(re[1]))
    cv2.circle(bgr, c1, eye_r, color, thickness, lineType=cv2.LINE_AA)
    cv2.circle(bgr, c2, eye_r, color, thickness, lineType=cv2.LINE_AA)
    bridge_y = int((le[1] + re[1]) / 2)
    cv2.line(
        bgr,
        (c1[0] + eye_r // 2, bridge_y),
        (c2[0] - eye_r // 2, bridge_y),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_mustache_overlay(bgr: np.ndarray, shape, color=(20, 20, 80)) -> None:
    """Filled ellipse over outer mouth contour."""
    mouth = np.array([(shape.part(i).x, shape.part(i).y) for i in range(48, 68)])
    x, y, w, h = cv2.boundingRect(mouth)
    if w < 2 or h < 2:
        return
    center = (x + w // 2, y + h // 2 + h // 6)
    axes = (max(w // 2, 3), max(h // 2, 2))
    cv2.ellipse(bgr, center, axes, 0, 0, 360, color, thickness=-1, lineType=cv2.LINE_AA)


def augment_facial_accessories(
    bgr: np.ndarray,
    detector: "dlib.fhog_object_detector",
    predictor: "dlib.shape_predictor",
    p_glasses: float = 0.35,
    p_mustache: float = 0.35,
) -> np.ndarray:
    """If a frontal face and landmarks are found, overlay random accessories."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rects = detector(rgb, 1)
    if len(rects) == 0:
        return bgr
    r = max(rects, key=lambda q: q.width() * q.height())
    shape = predictor(rgb, r)
    out = bgr.copy()
    if random.random() < p_glasses:
        draw_glasses_overlay(out, shape)
    if random.random() < p_mustache:
        draw_mustache_overlay(out, shape)
    return out


def load_dlib_facial_tools(
    shape_predictor_path: str | Path,
) -> tuple["dlib.fhog_object_detector", "dlib.shape_predictor"]:
    import dlib

    det = dlib.get_frontal_face_detector()
    pred = dlib.shape_predictor(str(shape_predictor_path))
    return det, pred
