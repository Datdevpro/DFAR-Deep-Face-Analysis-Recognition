"""
Augment images under ``data/raw/<employee_id>/*.jpg`` → ``data/aug/<employee_id>/``.

Usage:
  python scripts/augment_dataset.py --source data/raw --out data/aug --copies 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plan1_app.augmentation import augment_image
from plan1_app.config import SHAPE_PREDICTOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "aug")
    ap.add_argument("--copies", type=int, default=5, help="random augmentations per source image")
    args = ap.parse_args()

    if not SHAPE_PREDICTOR.is_file():
        print(f"Warning: {SHAPE_PREDICTOR} missing — stage 2 (accessories) skipped. Run scripts/download_models.py")

    random.seed(42)
    for person_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
        pid = person_dir.name
        dest_root = args.out / pid
        dest_root.mkdir(parents=True, exist_ok=True)
        imgs = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        for img_path in imgs:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            stem = img_path.stem
            for i in range(args.copies):
                out = augment_image(bgr)
                out_path = dest_root / f"{stem}_aug_{i:02d}.jpg"
                cv2.imwrite(str(out_path), out)
            print(f"{pid}: {img_path.name} → {args.copies} aug")


if __name__ == "__main__":
    main()
