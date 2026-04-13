"""Test one image with Plan1 recognizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plan1_app.pipeline.recognize import Plan1Recognizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "svm_plan1.joblib")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Missing image: {args.image}")
    if not args.artifact.is_file():
        raise FileNotFoundError(f"Missing artifact: {args.artifact}")

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f"Cannot read image: {args.image}")

    rec = Plan1Recognizer(svm_artifact=args.artifact, device=args.device)
    out = rec.recognize_frame(bgr)
    out.pop("embedding", None)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
