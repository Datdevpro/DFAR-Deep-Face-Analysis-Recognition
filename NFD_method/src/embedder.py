"""
ArcFace / InsightFace ONNX face embedder.

Pipeline:
  1. Receive aligned face chip (112×112 BGR from FaceAligner).
  2. Convert BGR → RGB.
  3. Normalise: pixel = (pixel - 127.5) / 128.0   (ArcFace standard).
  4. Transpose HWC → CHW, add batch dimension.
  5. Run ONNX inference → 512-D vector.
  6. L2-normalise → unit vector (required for cosine similarity via inner product).

Output shape: (512,) float32, L2-norm ≈ 1.0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FaceEmbedder:
    """
    ArcFace ONNX wrapper.

    Parameters
    ----------
    model_path:
        Path to ``arcface.onnx``.
    embedding_dim:
        Expected output dimension (default 512).  Used for validation.
    """

    def __init__(self, model_path: Path, embedding_dim: int = 512) -> None:
        import onnxruntime as ort  # type: ignore

        if not model_path.is_file():
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                "Download from InsightFace model zoo and place at models/arcface.onnx"
            )

        self._dim = embedding_dim
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Compute a 512-D L2-normalised embedding for one aligned face.

        Parameters
        ----------
        aligned_face:
            BGR uint8 numpy array of shape (112, 112, 3).

        Returns
        -------
        np.ndarray
            Shape (512,), dtype float32, L2-norm == 1.0.
        """
        blob = self._preprocess(aligned_face)
        raw = self._session.run([self._output_name], {self._input_name: blob})[0]
        vec = np.squeeze(raw).astype(np.float32)  # (512,)
        return self._l2_normalize(vec)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        """
        Convert BGR uint8 → float32 NCHW blob ready for ArcFace.

        Steps:
          BGR → RGB           (ArcFace models are trained on RGB)
          (x - 127.5) / 128  (model-specific normalisation)
          HWC → CHW           (channel-first for ONNX)
          add batch dim       (1, C, H, W)
        """
        rgb = face_bgr[:, :, ::-1]                         # BGR → RGB
        rgb = rgb.astype(np.float32)
        rgb = (rgb - 127.5) / 128.0                        # normalise
        chw = rgb.transpose(2, 0, 1)                       # HWC → CHW
        return chw[np.newaxis]                             # add batch → NCHW

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """
        L2-normalise so ||vec|| = 1.

        When both query and index vectors are L2-normalised, cosine similarity
        equals the inner product — allowing FAISS IndexFlatIP to serve as a
        cosine similarity search index.
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return vec  # zero vector — edge case
        return vec / norm
