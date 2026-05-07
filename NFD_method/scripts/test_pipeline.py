"""
Quick pipeline tests — verify each component independently before a full run.

Tests
-----
1. Detector on a static image → prints detected faces.
2. Embedder shape → must be (512,).
3. FAISS search with same-person images → top-1 score should be high.
4. Webcam smoke test → opens camera for 3 seconds then exits.

Usage
-----
python scripts/test_pipeline.py --image path/to/test.jpg [--skip-webcam]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aligner import FaceAligner
from src.config import CONFIG
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.utils import setup_logger
from src.vector_store import FaceVectorStore

logger = setup_logger("test_pipeline")

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


def test_detector(image_path: Path) -> bool:
    logger.info("── Test 1: FaceDetector ─────────────────────────────")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        logger.error(f"Cannot read image: {image_path}")
        return False
    try:
        det = FaceDetector(CONFIG.model.detector_path, CONFIG.detection)
        faces = det.detect(bgr)
        print(f"  Detected {len(faces)} face(s)")
        for i, f in enumerate(faces):
            lm_info = f"landmarks={'yes' if f.landmarks is not None else 'no'}"
            print(f"  [{i}] bbox={f.bbox.astype(int).tolist()}  score={f.score:.3f}  {lm_info}")
        ok = len(faces) > 0
        print(f"  {PASS if ok else FAIL} Detector test {'passed' if ok else 'failed (no faces)'}")
        return ok
    except Exception as exc:
        print(f"  {FAIL} {exc}")
        return False


def test_embedder(image_path: Path) -> bool:
    logger.info("── Test 2: FaceEmbedder shape = (512,) ──────────────")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return False
    try:
        det  = FaceDetector(CONFIG.model.detector_path, CONFIG.detection)
        aln  = FaceAligner()
        emb  = FaceEmbedder(CONFIG.model.embedder_path, CONFIG.identification.embedding_dim)

        faces = det.detect(bgr)
        if not faces:
            print(f"  {FAIL} No face found in test image")
            return False

        aligned = aln.align(bgr, faces[0])
        vec = emb.embed(aligned)

        expected_dim = CONFIG.identification.embedding_dim
        ok = vec.shape == (expected_dim,)
        norm = float(np.linalg.norm(vec))
        print(f"  shape={vec.shape}  L2-norm={norm:.6f}")
        print(f"  {PASS if ok else FAIL} Embedding shape {'matches' if ok else 'MISMATCH'} expected ({expected_dim},)")
        if abs(norm - 1.0) > 0.01:
            print(f"  {FAIL} L2-norm {norm:.4f} != 1.0 — check _l2_normalize in embedder.py")
            return False
        return ok
    except Exception as exc:
        print(f"  {FAIL} {exc}")
        return False


def test_faiss_search(image_path: Path) -> bool:
    logger.info("── Test 3: FAISS search (same-image round-trip) ─────")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return False
    try:
        det  = FaceDetector(CONFIG.model.detector_path, CONFIG.detection)
        aln  = FaceAligner()
        emb  = FaceEmbedder(CONFIG.model.embedder_path, CONFIG.identification.embedding_dim)
        store = FaceVectorStore(CONFIG.identification.embedding_dim)

        faces = det.detect(bgr)
        if not faces:
            print(f"  {FAIL} No face detected")
            return False

        vec = emb.embed(aln.align(bgr, faces[0]))

        # Enroll the same embedding under a test ID
        store.add_embeddings("TEST001", "Test Person", [vec], [str(image_path)])

        # Search with the same vector → score should be ≈ 1.0
        results = store.search(vec, top_k=5)
        score = results[0].score if results else 0.0
        ok = score > 0.99
        print(f"  top-1 score={score:.6f}  employee_id={results[0].employee_id if results else 'N/A'}")
        print(f"  {PASS if ok else FAIL} Same-image score {'≈ 1.0' if ok else '< 0.99 — unexpected'}")
        return ok
    except Exception as exc:
        print(f"  {FAIL} {exc}")
        return False


def test_webcam(duration: int = 3) -> bool:
    logger.info(f"── Test 4: Webcam smoke test ({duration}s) ─────────────")
    try:
        import time
        cap = cv2.VideoCapture(CONFIG.webcam_index)
        if not cap.isOpened():
            print(f"  {FAIL} Cannot open camera index={CONFIG.webcam_index}")
            return False
        start = time.perf_counter()
        frames = 0
        while time.perf_counter() - start < duration:
            ret, _ = cap.read()
            if ret:
                frames += 1
        cap.release()
        fps = frames / duration
        ok = frames > 0
        print(f"  Captured {frames} frames in {duration}s  ({fps:.1f} fps)")
        print(f"  {PASS if ok else FAIL} Webcam {'OK' if ok else 'failed'}")
        return ok
    except Exception as exc:
        print(f"  {FAIL} {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Test NFD pipeline components.")
    ap.add_argument("--image", type=Path, default=None,
                    help="Path to a test image with one clear face.")
    ap.add_argument("--skip-webcam", action="store_true",
                    help="Skip the webcam test.")
    args = ap.parse_args()

    results: dict[str, bool] = {}

    if args.image and args.image.is_file():
        results["detector"]  = test_detector(args.image)
        results["embedder"]  = test_embedder(args.image)
        results["faiss"]     = test_faiss_search(args.image)
    else:
        logger.warning("No --image provided; skipping detector/embedder/FAISS tests.")

    if not args.skip_webcam:
        results["webcam"] = test_webcam(duration=3)

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        all_pass = all_pass and ok
    print("=" * 50)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
