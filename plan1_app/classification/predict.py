"""Inference: SVM margin → label + score (not a calibrated probability)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plan1_app.classification.train_svm import load_artifact


class SVMPredictor:
    def __init__(self, artifact_path: str | Path, reject_threshold: float | None = None) -> None:
        self._pipe, self._classes, meta = load_artifact(Path(artifact_path))
        self.reject_threshold = (
            reject_threshold if reject_threshold is not None else float(meta.get("threshold", 0.0))
        )

    def predict_embedding(self, emb: np.ndarray) -> tuple[str | None, float, str]:
        """
        Returns ``(employee_id_or_none, score, note)``.
        ``score`` is the multiclass margin used for auditing (higher = more confident).
        """
        x = emb.reshape(1, -1)
        pred = self._pipe.predict(x)[0]
        df = np.asarray(self._pipe.decision_function(x))
        classes = np.asarray(self._classes)
        if df.ndim == 1 or df.shape[-1] == 1:
            raw = float(np.ravel(df)[0])
            if len(classes) == 2 and str(pred) == str(classes[1]):
                score = raw
            elif len(classes) == 2:
                score = -raw
            else:
                score = abs(raw)
        else:
            ci = int(np.flatnonzero(classes == pred)[0])
            score = float(np.atleast_2d(df)[0, ci])
        label = str(pred)
        if score < self.reject_threshold:
            return None, score, "below_threshold"
        return label, score, "ok"
