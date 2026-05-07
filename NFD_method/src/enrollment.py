"""
Employee enrollment pipeline.

For each image:
  1. Load BGR image from disk.
  2. Detect faces (SCRFD).
  3. Reject if 0 or >1 face detected.
  4. Run quality gate.
  5. Align face → 112×112.
  6. Extract 512-D ArcFace embedding.
  7. L2-normalise (done inside FaceEmbedder).
  8. Add to FAISS vector store.
  9. Save updated index to disk.

Usage (CLI wrapper is scripts/enroll_employee.py):
  pipeline = EnrollmentPipeline.from_config(CONFIG)
  result = pipeline.enroll_employee("EMP001", "Nguyen Van A", Path("data/employees/EMP001"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.aligner import FaceAligner
from src.config import AppConfig, CONFIG
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.quality import QualityChecker
from src.utils import setup_logger
from src.vector_store import FaceVectorStore

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
logger = setup_logger("enrollment")


@dataclass
class EnrollmentReport:
    """Summary of one employee's enrollment run."""
    employee_id: str
    name: str
    total_images: int
    accepted: int
    rejected_no_face: int
    rejected_multi_face: int
    rejected_quality: int
    rejected_embed_fail: int
    added_vectors: int
    rejection_details: list[str] = field(default_factory=list)


class EnrollmentPipeline:
    """
    Full enrollment pipeline for one employee.

    Parameters
    ----------
    detector:   FaceDetector instance.
    aligner:    FaceAligner instance.
    embedder:   FaceEmbedder instance.
    quality:    QualityChecker instance.
    store:      FaceVectorStore instance (already loaded or empty).
    config:     AppConfig for path resolution.
    """

    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: FaceEmbedder,
        quality: QualityChecker,
        store: FaceVectorStore,
        config: AppConfig,
    ) -> None:
        self._det = detector
        self._aln = aligner
        self._emb = embedder
        self._qlt = quality
        self._store = store
        self._cfg = config

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: AppConfig = CONFIG) -> "EnrollmentPipeline":
        """Build pipeline from the global AppConfig."""
        store = FaceVectorStore(config.identification.embedding_dim)

        # Load existing index if present (incremental enrollment)
        idx_path  = config.data.faiss_index_path
        meta_path = config.data.metadata_path
        if idx_path.is_file() and meta_path.is_file():
            store.load(idx_path, meta_path)
            logger.info(
                f"Loaded existing index: {store.total_vectors} vectors, "
                f"{len(store.enrolled_employees)} employees."
            )

        return cls(
            detector=FaceDetector(config.model.detector_path, config.detection),
            aligner=FaceAligner(output_size=112),
            embedder=FaceEmbedder(config.model.embedder_path, config.identification.embedding_dim),
            quality=QualityChecker(config.quality),
            store=store,
            config=config,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def enroll_employee(
        self,
        employee_id: str,
        name: str,
        image_dir: Path,
    ) -> EnrollmentReport:
        """
        Enroll an employee from all images in *image_dir*.

        Parameters
        ----------
        employee_id:
            Unique ID, e.g. "EMP001".
        name:
            Display name, e.g. "Nguyen Van A".
        image_dir:
            Directory containing face images (jpg/png/…).

        Returns
        -------
        EnrollmentReport
        """
        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        report = EnrollmentReport(
            employee_id=employee_id,
            name=name,
            total_images=len(images),
            accepted=0,
            rejected_no_face=0,
            rejected_multi_face=0,
            rejected_quality=0,
            rejected_embed_fail=0,
            added_vectors=0,
        )

        embeddings: list[np.ndarray] = []
        image_paths: list[str] = []

        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                report.rejected_embed_fail += 1
                report.rejection_details.append(f"{img_path.name}: cannot read")
                continue

            # ── 1. Detect ──────────────────────────────────────────────────────
            detections = self._det.detect(bgr)

            if len(detections) == 0:
                report.rejected_no_face += 1
                report.rejection_details.append(f"{img_path.name}: no face detected")
                logger.debug(f"  SKIP {img_path.name} — no face")
                continue

            if len(detections) > 1:
                report.rejected_multi_face += 1
                report.rejection_details.append(
                    f"{img_path.name}: {len(detections)} faces (expected 1)"
                )
                logger.debug(f"  SKIP {img_path.name} — {len(detections)} faces")
                continue

            det = detections[0]

            # ── 2. Quality gate ────────────────────────────────────────────────
            quality = self._qlt.check(bgr, det)
            if not quality.ok:
                report.rejected_quality += 1
                report.rejection_details.append(
                    f"{img_path.name}: quality={quality.reasons}"
                )
                logger.debug(f"  SKIP {img_path.name} — quality {quality.reasons}")
                continue

            # ── 3. Align ───────────────────────────────────────────────────────
            aligned = self._aln.align(bgr, det)

            # ── 4. Embed ───────────────────────────────────────────────────────
            try:
                vec = self._emb.embed(aligned)
            except Exception as exc:
                report.rejected_embed_fail += 1
                report.rejection_details.append(f"{img_path.name}: embed failed ({exc})")
                logger.warning(f"  FAIL {img_path.name} — {exc}")
                continue

            embeddings.append(vec)
            image_paths.append(str(img_path))
            report.accepted += 1
            logger.info(
                f"  OK  {img_path.name}  quality={quality.score:.2f}  "
                f"det_score={det.score:.2f}"
            )

        # ── 5. Add to store ────────────────────────────────────────────────────
        if embeddings:
            added = self._store.add_embeddings(
                employee_id=employee_id,
                name=name,
                embeddings=embeddings,
                image_paths=image_paths,
            )
            report.added_vectors = added

            # Persist after every employee enrollment
            self._save_index()
        else:
            logger.warning(f"No valid embeddings for {employee_id} — index not updated.")

        return report

    def get_store(self) -> FaceVectorStore:
        return self._store

    # ── Internal ───────────────────────────────────────────────────────────────

    def _save_index(self) -> None:
        idx_path  = self._cfg.data.faiss_index_path
        meta_path = self._cfg.data.metadata_path
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        self._store.save(idx_path, meta_path)
        logger.info(
            f"Index saved → {idx_path}  "
            f"({self._store.total_vectors} vectors, "
            f"{len(self._store.enrolled_employees)} employees)"
        )
