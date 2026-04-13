"""128-D embedding via dlib's ResNet face recognition model (FaceNet-class objective)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from plan1_app.config import CROP_SIZE, FACE_REC_MODEL, SHAPE_PREDICTOR


class DlibEmbedder:
    def __init__(
        self,
        shape_predictor_path: str | Path | None = None,
        face_rec_model_path: str | Path | None = None,
        chip_size: int = CROP_SIZE,
        padding: float = 0.25,
    ) -> None:
        import dlib

        sp = Path(shape_predictor_path or SHAPE_PREDICTOR)
        fr = Path(face_rec_model_path or FACE_REC_MODEL)
        if not sp.is_file():
            raise FileNotFoundError(f"Missing shape predictor: {sp}")
        if not fr.is_file():
            raise FileNotFoundError(f"Missing face recognition model: {fr}")

        self._dlib = dlib
        self._predictor = dlib.shape_predictor(str(sp))
        self._face_model = dlib.face_recognition_model_v1(str(fr))
        self._chip_size = chip_size
        self._padding = padding

    def embed_from_box(self, bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """
        ``box`` = [x1,y1,x2,y2]. Returns L2-normalized 128-D vector or None if landmarks fail.
        """
        dlib = self._dlib
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        shape = self._predictor(rgb, rect)
        try:
            detail = dlib.get_face_chip_details(shape, self._chip_size, self._padding)
            chip = dlib.extract_image_chip(rgb, detail)
        except RuntimeError:
            return None
        vec = np.array(self._face_model.compute_face_descriptor(chip), dtype=np.float64)
        n = np.linalg.norm(vec)
        if n < 1e-9:
            return None
        return (vec / n).astype(np.float32)
