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
import torch
from sklearn.metrics import accuracy_score, hinge_loss
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


def describe_device(requested: str | None) -> tuple[str, str]:
    """Return (resolved_device, human_readable_info)."""
    want = (requested or "").strip().lower()
    if want == "cuda":
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return "cuda", f"cuda:{idx} ({name})"
        return "cpu", "cpu (CUDA requested but not available)"
    return "cpu", "cpu"


def render_progress(done: int, total: int, ok: int, skip_no_face: int, skip_embed_fail: int) -> str:
    width = 26
    pct = int((done / total) * 100) if total else 100
    filled = int(width * done / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {pct:3d}% ({done}/{total}) "
        f"ok={ok} skip_no_face={skip_no_face} skip_embed={skip_embed_fail}"
    )


def try_hinge_loss(y_true: np.ndarray, decision: np.ndarray, classes: np.ndarray) -> float | None:
    try:
        if decision.ndim == 1:
            return float(hinge_loss(y_true, decision))
        return float(hinge_loss(y_true, decision, labels=list(classes)))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "svm_plan1.joblib")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    resolved_device, device_info = describe_device(args.device)
    det = Plan1CascadeDetector(device=resolved_device)
    emb = DlibEmbedder()

    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    import cv2

    labeled = list(iter_labeled_images(args.data))
    classes_in_dir = sorted({label for label, _ in labeled})
    print("== Training Setup ==")
    print(f"data_root={args.data}")
    print(f"artifact={args.artifact}")
    print(f"requested_device={args.device or 'auto'}")
    print(f"resolved_device={resolved_device}")
    print(f"device_info={device_info}")
    print(f"dataset_images={len(labeled)}")
    print(f"dataset_classes={len(classes_in_dir)}")
    if classes_in_dir:
        preview = ", ".join(classes_in_dir[:10])
        suffix = " ..." if len(classes_in_dir) > 10 else ""
        print(f"class_preview={preview}{suffix}")
    print("== Embedding Extraction ==")

    ok = 0
    skip_no_face = 0
    skip_embed_fail = 0
    total = len(labeled)
    for i, (label, path) in enumerate(labeled, start=1):
        bgr = cv2.imread(str(path))
        if bgr is None:
            skip_embed_fail += 1
            print("\r" + render_progress(i, total, ok, skip_no_face, skip_embed_fail), end="", flush=True)
            continue
        box, _ = det.detect_largest(bgr)
        if box is None:
            skip_no_face += 1
            print("\r" + render_progress(i, total, ok, skip_no_face, skip_embed_fail), end="", flush=True)
            continue
        v = emb.embed_from_box(bgr, box)
        if v is None:
            skip_embed_fail += 1
            print("\r" + render_progress(i, total, ok, skip_no_face, skip_embed_fail), end="", flush=True)
            continue
        X_list.append(v)
        y_list.append(label)
        ok += 1
        print("\r" + render_progress(i, total, ok, skip_no_face, skip_embed_fail), end="", flush=True)
    print("")

    if len(X_list) < 4:
        print("Need more labeled face images (multiple IDs, multiple photos).")
        sys.exit(1)

    classes_after = sorted(set(y_list))
    print("== Dataset Summary ==")
    print(f"embeddings_kept={len(X_list)}")
    print(f"classes_with_embeddings={len(classes_after)}")
    print(f"skipped_no_face={skip_no_face}")
    print(f"skipped_embed_fail={skip_embed_fail}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    print("== Model Training ==")
    print(f"train_samples={len(X_tr)}")
    print(f"val_samples={len(X_val)}")
    pipe = train_linear_svm(X_tr, y_tr)
    thr = default_threshold_from_validation(pipe, X_val, y_val)

    pred_tr = pipe.predict(X_tr)
    pred_val = pipe.predict(X_val)
    acc_tr = float(accuracy_score(y_tr, pred_tr))
    acc_val = float(accuracy_score(y_val, pred_val))
    df_tr = np.asarray(pipe.decision_function(X_tr))
    df_val = np.asarray(pipe.decision_function(X_val))
    classes = np.asarray(pipe.named_steps["svc"].classes_)
    loss_tr = try_hinge_loss(y_tr, df_tr, classes)
    loss_val = try_hinge_loss(y_val, df_val, classes)

    print("== Metrics ==")
    print(f"train_accuracy={acc_tr:.4f}")
    print(f"val_accuracy={acc_val:.4f}")
    if loss_tr is not None:
        print(f"train_hinge_loss={loss_tr:.4f}")
    if loss_val is not None:
        print(f"val_hinge_loss={loss_val:.4f}")
    print(f"avg_accuracy={(acc_tr + acc_val) / 2.0:.4f}")
    if loss_tr is not None and loss_val is not None:
        print(f"avg_hinge_loss={(loss_tr + loss_val) / 2.0:.4f}")

    meta = {"threshold": thr, "train_samples": len(X_tr), "val_samples": len(X_val)}
    meta.update(
        {
            "requested_device": args.device or "auto",
            "resolved_device": resolved_device,
            "device_info": device_info,
            "dataset_images": len(labeled),
            "dataset_classes": len(classes_in_dir),
            "kept_embeddings": len(X_list),
            "classes_with_embeddings": len(classes_after),
            "skipped_no_face": skip_no_face,
            "skipped_embed_fail": skip_embed_fail,
            "train_accuracy": acc_tr,
            "val_accuracy": acc_val,
            "avg_accuracy": (acc_tr + acc_val) / 2.0,
        }
    )
    if loss_tr is not None:
        meta["train_hinge_loss"] = loss_tr
    if loss_val is not None:
        meta["val_hinge_loss"] = loss_val
    if loss_tr is not None and loss_val is not None:
        meta["avg_hinge_loss"] = (loss_tr + loss_val) / 2.0

    save_artifact(pipe, classes, args.artifact, meta=meta)
    print(f"saved {args.artifact} threshold={thr:.4f}")
    (args.artifact.parent / "svm_plan1.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
