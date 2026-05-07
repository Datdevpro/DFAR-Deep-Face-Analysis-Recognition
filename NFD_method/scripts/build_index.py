"""
Rebuild the FAISS index from scratch by re-enrolling all employees
found under data/employees/.

Useful when you want to:
  - Regenerate the index after changing the ArcFace model.
  - Clean up deleted employee records.
  - Change embedding dimension.

Usage
-----
python scripts/build_index.py [--data-dir data/employees]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.enrollment import EnrollmentPipeline
from src.utils import setup_logger
from src.vector_store import FaceVectorStore

logger = setup_logger("build_index")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rebuild FAISS index from all employee image directories."
    )
    ap.add_argument(
        "--data-dir", type=Path,
        default=CONFIG.data.employees_dir,
        help="Root directory containing per-employee sub-directories.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List discovered employees but do not write index.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.is_dir():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Discover employee directories
    emp_dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir())
    if not emp_dirs:
        logger.error(f"No employee sub-directories found in {args.data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(emp_dirs)} employee director(ies): {[d.name for d in emp_dirs]}")

    if args.dry_run:
        print("Dry run — no index written.")
        return

    # Build a fresh store (ignore any existing index)
    CONFIG_COPY = CONFIG  # we'll just overwrite on disk at the end
    fresh_store = FaceVectorStore(CONFIG.identification.embedding_dim)

    # Build pipeline without loading existing index
    from src.aligner import FaceAligner
    from src.detector import FaceDetector
    from src.embedder import FaceEmbedder
    from src.quality import QualityChecker

    pipeline = EnrollmentPipeline(
        detector=FaceDetector(CONFIG.model.detector_path, CONFIG.detection),
        aligner=FaceAligner(output_size=112),
        embedder=FaceEmbedder(CONFIG.model.embedder_path, CONFIG.identification.embedding_dim),
        quality=QualityChecker(CONFIG.quality),
        store=fresh_store,
        config=CONFIG,
    )

    total_added = 0
    for emp_dir in emp_dirs:
        employee_id = emp_dir.name
        # Try to read display name from name.txt if present, else use directory name
        name_file = emp_dir / "name.txt"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.is_file() else employee_id

        logger.info(f"Processing {employee_id} ({name}) …")
        report = pipeline.enroll_employee(employee_id, name, emp_dir)
        total_added += report.added_vectors
        logger.info(
            f"  {employee_id}: accepted={report.accepted}, "
            f"added={report.added_vectors}, "
            f"rejected={report.rejected_no_face + report.rejected_quality + report.rejected_embed_fail}"
        )

    store = pipeline.get_store()
    print("\n" + "=" * 50)
    print("  INDEX BUILD COMPLETE")
    print("=" * 50)
    print(f"  Employees enrolled : {len(store.enrolled_employees)}")
    print(f"  Total vectors      : {store.total_vectors}")
    print(f"  Index path         : {CONFIG.data.faiss_index_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
