"""
Build embeddings with Plan1 pipeline (MTCNN + dlib 128-D) and train linear SVM.

Dataset layout::
  data/raw/<employee_id>/*.jpg   (or use --data with same layout)

Usage::
  python scripts/download_models.py
  python scripts/train_svm.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plan1_app.detection.cascade import Plan1CascadeDetector
from plan1_app.embedding.dlib_embedder import DlibEmbedder
from plan1_app.classification.train_svm import (
    default_threshold_from_validation,
    save_artifact,
    train_linear_svm,
)


def iter_labeled_images(root: Path):
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        label = person.name
        for p in sorted(person.glob("*.jpg")) + sorted(person.glob("*.png")):
            yield label, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "svm_plan1.joblib")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    det = Plan1CascadeDetector(device=args.device)
    emb = DlibEmbedder()

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    import cv2

    for label, path in iter_labeled_images(args.data):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        box, _ = det.detect_largest(bgr)
        if box is None:
            print(f"skip no face: {path}")
            continue
        v = emb.embed_from_box(bgr, box)
        if v is None:
            print(f"skip embed fail: {path}")
            continue
        X_list.append(v)
        y_list.append(label)
        print(f"ok {label} {path.name}")

    if len(X_list) < 4:
        print("Need more labeled face images (multiple IDs, multiple photos).")
        sys.exit(1)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    pipe = train_linear_svm(X_tr, y_tr)
    thr = default_threshold_from_validation(pipe, X_val, y_val)
    meta = {"threshold": thr, "train_samples": len(X_tr), "val_samples": len(X_val)}
    classes = np.asarray(pipe.named_steps["svc"].classes_)
    save_artifact(pipe, classes, args.artifact, meta=meta)
    print(f"saved {args.artifact} threshold={thr:.4f}")
    (args.artifact.parent / "svm_plan1.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
