"""End-to-end Plan 1: cascade detect → dlib chip+embed → SVM."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plan1_app.classification.predict import SVMPredictor
from plan1_app.detection.cascade import Plan1CascadeDetector
from plan1_app.embedding.dlib_embedder import DlibEmbedder


class Plan1Recognizer:
    def __init__(
        self,
        svm_artifact: str | Path,
        device: str | None = None,
        reject_threshold: float | None = None,
        shape_predictor_path: str | Path | None = None,
        face_rec_model_path: str | Path | None = None,
    ) -> None:
        self.detector = Plan1CascadeDetector(device=device)
        self.embedder = DlibEmbedder(
            shape_predictor_path=shape_predictor_path,
            face_rec_model_path=face_rec_model_path,
        )
        self.svm = SVMPredictor(svm_artifact, reject_threshold=reject_threshold)

    def recognize_frame(self, bgr: np.ndarray) -> dict:
        box, det_prob = self.detector.detect_largest(bgr)
        if box is None:
            return {
                "ok": False,
                "reason": "no_face",
                "box": None,
                "det_prob": None,
                "employee_id": None,
                "margin": None,
                "note": None,
            }
        emb = self.embedder.embed_from_box(bgr, box)
        if emb is None:
            return {
                "ok": False,
                "reason": "embed_failed",
                "box": box.tolist(),
                "det_prob": det_prob,
                "employee_id": None,
                "margin": None,
                "note": None,
            }
        label, margin, note = self.svm.predict_embedding(emb)
        return {
            "ok": label is not None,
            "reason": note if label is None else "match",
            "box": box.tolist(),
            "det_prob": det_prob,
            "employee_id": label,
            "margin": margin,
            "note": note,
            "embedding": emb,
        }
