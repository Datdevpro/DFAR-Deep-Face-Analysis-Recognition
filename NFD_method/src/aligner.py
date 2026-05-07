"""
Face alignment — warps a detected face to a canonical 112×112 crop
using a similarity transform fitted to the 5 ArcFace landmarks.

Reference template (ArcFace / CosFace standard):
    Left eye, Right eye, Nose tip, Left mouth, Right mouth
    — normalised for a 112×112 output chip.
"""

from __future__ import annotations

import numpy as np
import cv2

from src.utils import FaceDetection

# ── ArcFace reference 5-point template (112×112) ─────────────────────────────
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceAligner:
    """
    Aligns a face to the ArcFace 112×112 canonical crop via affine warp.

    If 5 landmarks are available an accurate similarity transform is used.
    When landmarks are missing the face bbox is simply resized to 112×112
    (lower quality — enrolment with landmarks strongly recommended).
    """

    def __init__(self, output_size: int = 112) -> None:
        self._size = output_size

    # ── Public API ────────────────────────────────────────────────────────────

    def align(self, frame: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """
        Return an aligned BGR face chip of shape (output_size, output_size, 3).

        Parameters
        ----------
        frame:
            Full BGR frame.
        detection:
            FaceDetection; landmarks used if present.
        """
        if detection.landmarks is not None and len(detection.landmarks) == 5:
            return self._align_with_landmarks(frame, detection.landmarks)
        return self._align_bbox_fallback(frame, detection.bbox)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _align_with_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,  # (5, 2) float32
    ) -> np.ndarray:
        """
        Compute similarity transform M from detected landmarks → ARCFACE_DST,
        then warp the full frame so the face lands exactly on the template.

        cv2.estimateAffinePartial2D finds the best-fit (rotation + uniform
        scale + translation) — no shear — which preserves face geometry.
        """
        # Scale DST template to output_size (template is defined for 112)
        scale = self._size / 112.0
        dst = ARCFACE_DST * scale

        src = landmarks.astype(np.float32)  # (5, 2)

        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            # Fallback if estimation fails
            return self._align_bbox_fallback(frame, None)

        aligned = cv2.warpAffine(
            frame,
            M,
            (self._size, self._size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return aligned

    def _align_bbox_fallback(
        self,
        frame: np.ndarray,
        bbox: np.ndarray | None,
    ) -> np.ndarray:
        """Simple crop + resize fallback when landmarks are unavailable."""
        h, w = frame.shape[:2]
        if bbox is not None:
            x1, y1, x2, y2 = (int(max(0, v)) for v in bbox)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                return cv2.resize(crop, (self._size, self._size))
        # Ultimate fallback: black chip
        return np.zeros((self._size, self._size, 3), dtype=np.uint8)
