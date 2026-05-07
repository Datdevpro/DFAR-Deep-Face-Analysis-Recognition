"""Two-stage augmentation pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plan1_app.augmentation.stage1_general import augment_general
from plan1_app.augmentation.stage2_facial import augment_facial_accessories, load_dlib_facial_tools
from plan1_app.config import SHAPE_PREDICTOR


def augment_image(
    bgr: np.ndarray,
    *,
    shape_predictor_path: str | Path | None = None,
    dlib_detector=None,
    dlib_predictor=None,
    skip_facial_if_no_model: bool = True,
) -> np.ndarray:
    """
    Stage 1 (general) then optional stage 2 (facial accessories).
    If dlib models are missing and skip_facial_if_no_model is True, only stage 1 runs.
    """
    path = Path(shape_predictor_path or SHAPE_PREDICTOR)
    out = augment_general(bgr)
    if dlib_detector is not None and dlib_predictor is not None:
        return augment_facial_accessories(out, dlib_detector, dlib_predictor)
    if not path.is_file():
        if skip_facial_if_no_model:
            return out
        raise FileNotFoundError(f"Shape predictor not found: {path}")
    det, pred = load_dlib_facial_tools(path)
    return augment_facial_accessories(out, det, pred)
