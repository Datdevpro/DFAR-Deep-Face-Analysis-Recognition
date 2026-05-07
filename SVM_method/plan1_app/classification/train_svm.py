"""Train LinearSVM on embedding matrix."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def train_linear_svm(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
) -> Pipeline:
    """
    ``X`` shape (n, 128), ``y`` string or int labels.
    Scaler + LinearSVC (OV); decision_function used as confidence proxy at inference.
    """
    clf = LinearSVC(C=C, class_weight=class_weight, dual="auto", max_iter=10_000)
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", clf)])
    pipe.fit(X, y)
    return pipe


def save_artifact(pipe: Pipeline, classes: np.ndarray, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "classes": classes, "meta": meta or {}}, path)


def load_artifact(path: Path) -> tuple[Pipeline, np.ndarray, dict]:
    blob = joblib.load(path)
    return blob["pipeline"], blob["classes"], blob.get("meta", {})


def margin_scores(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """Same margin definition as ``SVMPredictor`` (for threshold tuning)."""
    pred = pipe.predict(X)
    df = np.asarray(pipe.decision_function(X))
    classes = np.asarray(pipe.named_steps["svc"].classes_)
    scores = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        p = pred[i]
        if df.ndim == 1:
            raw = float(np.ravel(df[i])[0])
            if len(classes) == 2:
                scores[i] = raw if str(p) == str(classes[1]) else -raw
            else:
                scores[i] = abs(raw)
        else:
            ci = int(np.flatnonzero(classes == p)[0])
            row = df[i] if df.ndim > 1 else df
            scores[i] = float(np.ravel(row)[ci])
    return scores


def default_threshold_from_validation(
    pipe: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Minimum margin among correctly classified validation rows (conservative reject threshold)."""
    if len(X_val) == 0:
        return 0.0
    margins = margin_scores(pipe, X_val)
    correct = pipe.predict(X_val) == y_val
    if not np.any(correct):
        return 0.0
    return float(np.min(margins[correct]))
