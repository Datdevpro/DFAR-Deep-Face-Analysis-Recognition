"""
Plan 1 face detection — three-stage CNN cascade.

Each stage (P-Net, R-Net, O-Net) has a **face vs background classifier** and a
**bounding-box regression** head — six convolutional predictors in total. This
matches the common MTCNN design; weights ship with ``facenet-pytorch``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class Plan1CascadeDetector:
    """Thin wrapper around pretrained MTCNN (six-head cascade)."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        min_face_size: int = 40,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7),
        factor: float = 0.709,
    ) -> None:
        from facenet_pytorch import MTCNN

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self._mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=factor,
            post_process=False,
            keep_all=True,
            device=self.device,
        )

    def detect_largest(self, bgr: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        """
        Returns ``(box, prob)`` where ``box`` is ``[x1, y1, x2, y2]`` in pixel coords,
        or ``(None, None)`` if no face.
        """
        rgb = bgr[:, :, ::-1].copy()
        out = self._mtcnn.detect(rgb, landmarks=False)
        boxes, probs = out[0], out[1]
        if boxes is None or len(boxes) == 0:
            return None, None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        j = int(np.argmax(areas))
        return boxes[j].astype(np.float32), float(probs[j])

    def detect_all(self, bgr: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """All boxes sorted by descending area."""
        rgb = bgr[:, :, ::-1].copy()
        out = self._mtcnn.detect(rgb, landmarks=False)
        boxes, probs = out[0], out[1]
        if boxes is None or len(boxes) == 0:
            return [], []
        order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        return [boxes[i].astype(np.float32) for i in order], [float(probs[i]) for i in order]
