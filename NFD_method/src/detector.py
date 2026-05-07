"""
Face detector — wraps SCRFD ONNX via insightface.

The public interface is FaceDetector.detect(frame) which returns a list of
FaceDetection objects.  Swapping the underlying implementation only requires
editing this file.

Implementation notes
--------------------
* Uses insightface.model_zoo.scrfd.SCRFD for output decoding (handles the
  multi-stride anchor grid automatically).
* Falls back to raw onnxruntime if insightface is unavailable (limited — no
  landmark support in fallback).
* Always converts BGR→RGB internally before inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import DetectionConfig
from src.utils import FaceDetection


class FaceDetector:
    """
    SCRFD-based face detector.

    Parameters
    ----------
    model_path:
        Path to ``scrfd.onnx``.
    config:
        DetectionConfig (thresholds, input size, …).
    """

    def __init__(self, model_path: Path, config: DetectionConfig) -> None:
        self._cfg = config
        self._backend: str

        if not model_path.is_file():
            raise FileNotFoundError(
                f"SCRFD model not found: {model_path}\n"
                "Download from InsightFace model zoo and place at models/scrfd.onnx"
            )

        # ── Try insightface SCRFD loader (preferred) ──────────────────────────
        try:
            from insightface.model_zoo.scrfd import SCRFD  # type: ignore

            self._model = SCRFD(model_file=str(model_path))
            # ctx_id: -1 = CPU, 0 = first GPU
            self._model.prepare(ctx_id=-1, input_size=config.input_size)
            self._backend = "insightface"
        except Exception as exc:
            # ── Fallback: raw onnxruntime ─────────────────────────────────────
            import onnxruntime as ort  # type: ignore

            print(
                f"[FaceDetector] insightface SCRFD unavailable ({exc}); "
                "falling back to raw onnxruntime (no landmarks)."
            )
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._backend = "onnxruntime_raw"

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Detect all faces in *frame* (BGR uint8).

        Returns a list of FaceDetection sorted by descending detector score.
        """
        if self._backend == "insightface":
            return self._detect_insightface(frame)
        return self._detect_raw(frame)

    # ── Backends ──────────────────────────────────────────────────────────────

    def _detect_insightface(self, frame: np.ndarray) -> list[FaceDetection]:
        """Use insightface SCRFD wrapper."""
        # insightface expects BGR; detect() returns (bboxes, kpss)
        bboxes, kpss = self._model.detect(
            frame,
            thresh=self._cfg.conf_threshold,
            input_size=self._cfg.input_size,
        )
        if bboxes is None or len(bboxes) == 0:
            return []

        results: list[FaceDetection] = []
        for i, box in enumerate(bboxes[: self._cfg.max_faces]):
            x1, y1, x2, y2, score = box
            lm: Optional[np.ndarray] = None
            if kpss is not None and i < len(kpss):
                lm = kpss[i].astype(np.float32)  # (5, 2)
            results.append(
                FaceDetection(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    score=float(score),
                    landmarks=lm,
                )
            )
        # Sort by descending score
        results.sort(key=lambda d: d.score, reverse=True)
        return results

    def _detect_raw(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Minimal raw onnxruntime fallback (no landmark support).
        Works for simple SCRFD variants with flat output layout.
        """
        ih, iw = self._cfg.input_size
        blob = cv2.resize(frame, (iw, ih))
        blob = (blob.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW

        outs = self._session.run(None, {self._input_name: blob})

        # Heuristic: assume first output is score, second is bbox
        # (model-specific — works for some single-stride variants)
        results: list[FaceDetection] = []
        if len(outs) >= 2:
            scores_raw = np.squeeze(outs[0])
            bboxes_raw = np.squeeze(outs[1])
            if scores_raw.ndim == 0:
                return results
            mask = scores_raw > self._cfg.conf_threshold
            for score, box in zip(scores_raw[mask], bboxes_raw[mask]):
                # Scale back to original frame size
                scale_x = frame.shape[1] / iw
                scale_y = frame.shape[0] / ih
                x1, y1, x2, y2 = (
                    box[0] * scale_x, box[1] * scale_y,
                    box[2] * scale_x, box[3] * scale_y,
                )
                results.append(
                    FaceDetection(
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        score=float(score),
                    )
                )
        results.sort(key=lambda d: d.score, reverse=True)
        return results[: self._cfg.max_faces]
