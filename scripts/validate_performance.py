"""
Validate trained SVM model and plot performance reports.

For this project (Linear SVM), there is no epoch-by-epoch training history like deep
neural nets. This script evaluates train/val split metrics and saves charts:
  - accuracy/loss comparison (train vs val)
  - confusion matrix (validation set)
  - margin score distribution

Usage:
  python scripts/validate_performance.py --artifact artifacts/svm_plan1.joblib --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, hinge_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plan1_app.classification.train_svm import load_artifact
from plan1_app.detection.cascade import Plan1CascadeDetector
from plan1_app.embedding.dlib_embedder import DlibEmbedder


def _safe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc
    return plt


def iter_labeled_images(root: Path):
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        label = person.name
        for p in sorted(person.glob("*.jpg")) + sorted(person.glob("*.png")):
            yield label, p


def describe_device(requested: str | None) -> str:
    import torch

    want = (requested or "").strip().lower()
    if want == "cuda":
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} ({name})"
        return "cpu (CUDA requested but unavailable)"
    return "cpu"


def extract_embeddings(data_root: Path, device: str | None):
    detector = Plan1CascadeDetector(device=device)
    embedder = DlibEmbedder()

    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    skipped_no_face = 0
    skipped_embed_fail = 0

    labeled = list(iter_labeled_images(data_root))
    for label, path in labeled:
        bgr = cv2.imread(str(path))
        if bgr is None:
            skipped_embed_fail += 1
            continue
        box, _ = detector.detect_largest(bgr)
        if box is None:
            skipped_no_face += 1
            continue
        emb = embedder.embed_from_box(bgr, box)
        if emb is None:
            skipped_embed_fail += 1
            continue
        X_list.append(emb)
        y_list.append(label)

    if len(X_list) < 4:
        raise RuntimeError("Not enough valid embeddings extracted for validation.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, len(labeled), skipped_no_face, skipped_embed_fail


def safe_hinge_loss(y_true: np.ndarray, decision: np.ndarray, classes: np.ndarray) -> float | None:
    try:
        if decision.ndim == 1:
            return float(hinge_loss(y_true, decision))
        return float(hinge_loss(y_true, decision, labels=list(classes)))
    except Exception:
        return None


def plot_accuracy_loss(plt, out_path: Path, acc_tr: float, acc_val: float, loss_tr: float | None, loss_val: float | None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(["train", "val"], [acc_tr, acc_val], marker="o")
    axes[0].set_title("Accuracy (Train vs Val)")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)

    if loss_tr is None or loss_val is None:
        axes[1].text(0.5, 0.5, "Hinge loss unavailable", ha="center", va="center")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    else:
        axes[1].plot(["train", "val"], [loss_tr, loss_val], marker="o", color="#C44E52")
        axes[1].set_title("Hinge Loss (Train vs Val)")
        axes[1].set_ylabel("Loss")
        axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(plt, out_path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_margin_distribution(plt, out_path: Path, margins: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(margins, bins=30, alpha=0.85, color="#4C8DFF", edgecolor="black")
    ax.set_title("Decision Margin Distribution (Validation)")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "svm_plan1.joblib")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "validation")
    args = ap.parse_args()

    if not args.artifact.is_file():
        raise FileNotFoundError(f"Missing artifact: {args.artifact}")

    plt = _safe_import_matplotlib()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("== Validation Setup ==")
    print(f"data_root={args.data}")
    print(f"artifact={args.artifact}")
    print(f"device={describe_device(args.device)}")
    print(f"out_dir={args.out_dir}")

    X, y, total_images, skipped_no_face, skipped_embed_fail = extract_embeddings(args.data, args.device)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    pipe, classes, meta = load_artifact(args.artifact)
    pred_tr = pipe.predict(X_tr)
    pred_val = pipe.predict(X_val)
    acc_tr = float(accuracy_score(y_tr, pred_tr))
    acc_val = float(accuracy_score(y_val, pred_val))

    df_tr = np.asarray(pipe.decision_function(X_tr))
    df_val = np.asarray(pipe.decision_function(X_val))
    classes = np.asarray(classes)
    loss_tr = safe_hinge_loss(y_tr, df_tr, classes)
    loss_val = safe_hinge_loss(y_val, df_val, classes)

    if df_val.ndim == 1:
        margins_val = np.abs(np.ravel(df_val))
    else:
        pred_to_idx = {str(c): i for i, c in enumerate(classes)}
        margins_val = np.array([df_val[i, pred_to_idx[str(p)]] for i, p in enumerate(pred_val)], dtype=np.float64)

    acc_loss_path = args.out_dir / "accuracy_loss.png"
    cm_path = args.out_dir / "confusion_matrix_val.png"
    margins_path = args.out_dir / "margin_distribution_val.png"
    report_path = args.out_dir / "validation_report.json"

    plot_accuracy_loss(plt, acc_loss_path, acc_tr, acc_val, loss_tr, loss_val)
    plot_confusion_matrix(plt, cm_path, y_val, pred_val, classes)
    plot_margin_distribution(plt, margins_path, margins_val)

    report = {
        "data_root": str(args.data),
        "artifact": str(args.artifact),
        "dataset_images": int(total_images),
        "dataset_classes": int(len(np.unique(y))),
        "kept_embeddings": int(len(X)),
        "skipped_no_face": int(skipped_no_face),
        "skipped_embed_fail": int(skipped_embed_fail),
        "train_samples": int(len(X_tr)),
        "val_samples": int(len(X_val)),
        "train_accuracy": acc_tr,
        "val_accuracy": acc_val,
        "avg_accuracy": (acc_tr + acc_val) / 2.0,
        "train_hinge_loss": loss_tr,
        "val_hinge_loss": loss_val,
        "avg_hinge_loss": None if loss_tr is None or loss_val is None else (loss_tr + loss_val) / 2.0,
        "artifact_meta": meta,
        "plots": {
            "accuracy_loss": str(acc_loss_path),
            "confusion_matrix_val": str(cm_path),
            "margin_distribution_val": str(margins_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("== Validation Metrics ==")
    print(f"train_accuracy={acc_tr:.4f}")
    print(f"val_accuracy={acc_val:.4f}")
    if loss_tr is not None:
        print(f"train_hinge_loss={loss_tr:.4f}")
    if loss_val is not None:
        print(f"val_hinge_loss={loss_val:.4f}")
    print(f"saved_report={report_path}")
    print(f"saved_plot={acc_loss_path}")
    print(f"saved_plot={cm_path}")
    print(f"saved_plot={margins_path}")


if __name__ == "__main__":
    main()

