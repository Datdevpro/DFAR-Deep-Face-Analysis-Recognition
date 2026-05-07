"""
Download dlib 68-point shape predictor and 128-D ResNet face model into ``plan1/models/``.
"""

from __future__ import annotations

import bz2
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"

URLS = {
    "shape_predictor_68_face_landmarks.dat.bz2": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat.bz2": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
}


def fetch(name: str, url: str) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    bz = MODELS / name
    out = MODELS / name.replace(".bz2", "")
    if out.is_file():
        print(f"skip (exists): {out}")
        return out
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url, timeout=120) as r, open(bz, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"decompressing {bz} ...")
    with bz2.open(bz, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    bz.unlink(missing_ok=True)
    print(f"ok: {out}")
    return out


def main() -> None:
    for name, url in URLS.items():
        fetch(name, url)


if __name__ == "__main__":
    main()
