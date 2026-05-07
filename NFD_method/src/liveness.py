"""
Liveness / anti-spoofing checker.

Current status: STUB — always returns is_live=True.

The interface is designed so that the stub can be replaced with a real
ONNX anti-spoofing model (e.g., MiniFASNet, Silent-Face) without
changing any caller code.

To integrate a real model later:
  1. Load the ONNX session in __init__.
  2. Replace _stub_check() with real inference in check().
  3. Tune the `score_threshold` config value.
"""

from __future__ import annotations

import numpy as np

from src.utils import FaceDetection, LivenessResult


class LivenessChecker:
    """
    Anti-spoofing interface.

    Parameters
    ----------
    model_path:
        Path to an ONNX anti-spoofing model.  Currently unused (stub mode).
    score_threshold:
        Minimum liveness score to consider the face as live.
        Range 0.0–1.0.  Stub always returns 1.0.
    """

    def __init__(
        self,
        model_path: str | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        self._model_path = model_path
        self._threshold = score_threshold
        self._stub = True  # flip to False when real model is loaded

        if model_path is not None:
            # TODO: load ONNX session here
            # import onnxruntime as ort
            # self._session = ort.InferenceSession(model_path, ...)
            # self._stub = False
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        aligned_face: np.ndarray,
    ) -> LivenessResult:
        """
        Run liveness check on an aligned face crop.

        Parameters
        ----------
        frame:
            Full BGR frame from webcam.
        detection:
            FaceDetection result (contains bbox and landmarks).
        aligned_face:
            Aligned 112×112 BGR face crop (same input as the embedder).

        Returns
        -------
        LivenessResult
            is_live=True, score=1.0 in stub mode.
        """
        if self._stub:
            return self._stub_check()

        # TODO: real inference path
        return self._stub_check()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _stub_check() -> LivenessResult:
        """Placeholder — unconditionally accepts all faces as live."""
        return LivenessResult(is_live=True, score=1.0, reason="stub")
