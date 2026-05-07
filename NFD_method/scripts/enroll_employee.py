"""
Enroll an employee from a directory of face images.

Usage
-----
python scripts/enroll_employee.py \\
    --employee-id EMP001 \\
    --name "Nguyen Van A" \\
    --image-dir data/employees/EMP001

The script will:
  1. Load each image in the directory.
  2. Detect faces (SCRFD).
  3. Apply quality gate.
  4. Align and extract 512-D ArcFace embedding.
  5. Add to FAISS index (incremental — existing entries preserved).
  6. Print an enrollment report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG
from src.enrollment import EnrollmentPipeline
from src.utils import setup_logger

logger = setup_logger("enroll_script")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Enroll an employee into the FAISS face index."
    )
    ap.add_argument(
        "--employee-id", required=True,
        help='Unique employee ID, e.g. "EMP001".',
    )
    ap.add_argument(
        "--name", required=True,
        help='Display name, e.g. "Nguyen Van A".',
    )
    ap.add_argument(
        "--image-dir", required=True, type=Path,
        help="Directory containing face images (jpg/png/…).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_dir.is_dir():
        logger.error(f"Image directory not found: {args.image_dir}")
        sys.exit(1)

    logger.info(
        f"Enrolling employee_id={args.employee_id!r}  name={args.name!r}  "
        f"image_dir={args.image_dir}"
    )

    pipeline = EnrollmentPipeline.from_config(CONFIG)
    report = pipeline.enroll_employee(
        employee_id=args.employee_id,
        name=args.name,
        image_dir=args.image_dir,
    )

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  ENROLLMENT REPORT")
    print("=" * 50)
    print(f"  Employee ID   : {report.employee_id}")
    print(f"  Name          : {report.name}")
    print(f"  Total images  : {report.total_images}")
    print(f"  Accepted      : {report.accepted}")
    print(f"  Vectors added : {report.added_vectors}")
    print(f"  Rejected")
    print(f"    no face     : {report.rejected_no_face}")
    print(f"    multi face  : {report.rejected_multi_face}")
    print(f"    quality     : {report.rejected_quality}")
    print(f"    embed fail  : {report.rejected_embed_fail}")

    if report.rejection_details:
        print("\n  Rejection details:")
        for detail in report.rejection_details:
            print(f"    - {detail}")

    store = pipeline.get_store()
    print(f"\n  Index total   : {store.total_vectors} vectors")
    print(f"  Employees     : {', '.join(sorted(store.enrolled_employees))}")
    print("=" * 50)

    if report.added_vectors == 0:
        logger.warning("No vectors were added.  Check image quality and model files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
