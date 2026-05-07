"""
Download and prepare a sample face-recognition dataset in Plan1 layout.

Output layout:
  data/raw/<identity>/*.jpg

Default source uses sklearn's LFW mirror and keeps identities with enough images.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def slugify_identity(name: str) -> str:
    """Convert identity names to folder-safe labels."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    return cleaned.strip("_").lower() or "unknown"


def to_uint8_hwc(arr: np.ndarray) -> np.ndarray:
    """
    sklearn ``fetch_lfw_people`` images are often float32 in [0, 1].
    Naive ``astype(uint8)`` turns them into black frames — convert safely to uint8 HxWxC.
    """
    x = np.asarray(arr)
    if x.dtype != np.uint8:
        x = x.astype(np.float64, copy=False)
        xmax = float(x.max()) if x.size else 0.0
        if xmax <= 1.0:
            x = x * 255.0
        x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x


def save_images(
    images: np.ndarray,
    target_names: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    per_identity_limit: int,
    image_size: int,
    dry_run: bool,
) -> tuple[int, int]:
    kept_identities = 0
    written_images = 0

    for identity_idx, raw_name in enumerate(target_names):
        indices = np.where(labels == identity_idx)[0]
        if len(indices) == 0:
            continue

        identity = slugify_identity(str(raw_name))
        kept_identities += 1
        picked = indices[:per_identity_limit] if per_identity_limit > 0 else indices

        if dry_run:
            written_images += len(picked)
            continue

        person_dir = out_dir / identity
        person_dir.mkdir(parents=True, exist_ok=True)

        for j, img_idx in enumerate(picked):
            arr = to_uint8_hwc(images[img_idx])
            pil = Image.fromarray(arr)
            if image_size > 0 and pil.size != (image_size, image_size):
                pil = pil.resize((image_size, image_size), Image.BILINEAR)
            out_path = person_dir / f"{j:04d}.jpg"
            pil.save(out_path, format="JPEG", quality=95)
            written_images += 1

    return kept_identities, written_images


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download LFW sample dataset and export to data/raw/<identity>/*.jpg"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument(
        "--min-faces-per-identity",
        type=int,
        default=20,
        help="Keep identities with at least this many images.",
    )
    ap.add_argument(
        "--per-identity-limit",
        type=int,
        default=40,
        help="Max images written per identity (0 = no limit).",
    )
    ap.add_argument(
        "--resize",
        type=int,
        default=160,
        help="Resize output images to square size (0 = keep original).",
    )
    ap.add_argument(
        "--color",
        action="store_true",
        help="Use RGB images (default is grayscale from sklearn).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating files.",
    )
    args = ap.parse_args()

    print("Downloading LFW metadata/images (first run may take time)...")
    data = fetch_lfw_people(
        data_home=str(ROOT / "data" / "external"),
        min_faces_per_person=args.min_faces_per_identity,
        color=args.color,
        resize=1.0,
        download_if_missing=True,
    )

    images = data.images
    if images.ndim == 4 and images.shape[-1] == 3:
        # Keep RGB
        pass
    elif images.ndim == 3:
        # Grayscale -> RGB for downstream consistency.
        images = np.stack([images, images, images], axis=-1)
    else:
        raise RuntimeError(f"Unexpected image tensor shape: {images.shape}")

    kept, written = save_images(
        images=images,
        target_names=data.target_names,
        labels=data.target,
        out_dir=args.out,
        per_identity_limit=args.per_identity_limit,
        image_size=args.resize,
        dry_run=args.dry_run,
    )

    mode = "dry-run" if args.dry_run else "done"
    print(f"{mode}: identities={kept}, images={written}, out={args.out}")


if __name__ == "__main__":
    main()

