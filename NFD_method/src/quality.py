"""
Face quality gate.

Checks a detected face against multiple quality criteria and returns
a QualityResult with an aggregate score and a list of failure reasons.

Checks performed:
  1. face_too_small  — bbox width or height < min_face_size
  2. blur            — variance of Laplacian < blur_threshold
  3. too_dark        — mean grayscale brightness < min_brightness
  4. too_bright      — mean grayscale brightness > max_brightness
  5. near_edge       — face bbox clips the frame edge by > edge_margin_ratio
  6. bad_landmarks   — inter-eye distance < min_eye_distance (if landmarks present)
"""

from __future__ import annotations

import numpy as np
import cv2

from src.config import QualityConfig
from src.utils import FaceDetection, QualityResult


class QualityChecker:
    """
    Stateless quality gate.  Instantiate once and call .check() per face.
    """

    def __init__(self, config: QualityConfig) -> None:
        self._cfg = config

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> QualityResult:
        """
        Run all quality checks on the face region inside *frame*.

        Parameters
        ----------
        frame:
            Full BGR frame from webcam / disk.
        detection:
            FaceDetection with bbox and optional landmarks.

        Returns
        -------
        QualityResult
            ok=True only if all mandatory checks pass.
        """
        reasons: list[str] = []
        scores: list[float] = []

        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        fh, fw = frame.shape[:2]

        # Clamp bbox to frame
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(fw - 1, x2), min(fh - 1, y2)

        # ── 1. Face size ──────────────────────────────────────────────────────
        face_w = x2 - x1
        face_h = y2 - y1
        min_dim = min(face_w, face_h)
        size_score = min(1.0, min_dim / (self._cfg.min_face_size * 2))
        scores.append(size_score)
        if min_dim < self._cfg.min_face_size:
            reasons.append("face_too_small")

        # Extract face ROI for remaining checks
        face_roi = frame[y1c:y2c, x1c:x2c]
        if face_roi.size == 0:
            # Cannot crop — fail all
            return QualityResult(ok=False, score=0.0, reasons=["invalid_bbox"])

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # ── 2. Blur (variance of Laplacian) ───────────────────────────────────
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = min(1.0, lap_var / (self._cfg.blur_threshold * 3))
        scores.append(blur_score)
        if lap_var < self._cfg.blur_threshold:
            reasons.append("blur")

        # ── 3 & 4. Brightness ────────────────────────────────────────────────
        mean_bright = float(gray.mean())
        # normalise: 0 at extremes, 1 at ideal (130)
        bright_score = 1.0 - abs(mean_bright - 130) / 130.0
        bright_score = float(np.clip(bright_score, 0.0, 1.0))
        scores.append(bright_score)
        if mean_bright < self._cfg.min_brightness:
            reasons.append("too_dark")
        elif mean_bright > self._cfg.max_brightness:
            reasons.append("too_bright")

        # ── 5. Near-edge check ────────────────────────────────────────────────
        margin_x = int(fw * self._cfg.edge_margin_ratio)
        margin_y = int(fh * self._cfg.edge_margin_ratio)
        near_edge = (x1 < margin_x or y1 < margin_y or
                     x2 > fw - margin_x or y2 > fh - margin_y)
        edge_score = 0.0 if near_edge else 1.0
        scores.append(edge_score)
        if near_edge:
            reasons.append("near_edge")

        # ── 6. Landmark sanity ────────────────────────────────────────────────
        if detection.landmarks is not None and len(detection.landmarks) == 5:
            left_eye  = detection.landmarks[0]   # standard order: left, right, nose, left_mouth, right_mouth
            right_eye = detection.landmarks[1]
            eye_dist = float(np.linalg.norm(right_eye - left_eye))
            lm_score = min(1.0, eye_dist / (self._cfg.min_eye_distance * 2))
            scores.append(lm_score)
            if eye_dist < self._cfg.min_eye_distance:
                reasons.append("bad_landmarks")

        aggregate = float(np.mean(scores)) if scores else 0.0
        ok = len(reasons) == 0
        return QualityResult(ok=ok, score=round(aggregate, 3), reasons=reasons)
