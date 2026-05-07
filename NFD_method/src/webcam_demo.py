"""
Real-time webcam identification demo.

Controls:
  q  — quit
  s  — save current frame to debug/

Decision logic (1:N identification):
  ACCEPT         top1_score >= ACCEPT_THRESHOLD
                 AND top1_score - top2_score >= MARGIN_THRESHOLD
                 AND quality_ok AND liveness_ok

  LOW_QUALITY    quality gate failed
  SPOOF_SUSPECTED liveness check failed
  UNKNOWN        does not meet ACCEPT conditions
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.aligner import FaceAligner
from src.config import AppConfig, CONFIG
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.liveness import LivenessChecker
from src.quality import QualityChecker
from src.utils import (
    FaceDetection,
    IdentificationResult,
    LivenessResult,
    QualityResult,
    SearchResult,
    Timer,
    draw_face,
    draw_hud,
    draw_status,
    log_identification,
    setup_logger,
)
from src.vector_store import FaceVectorStore

logger = setup_logger("webcam")


class WebcamDemo:
    """
    Real-time face identification demo.

    Parameters
    ----------
    config:
        Global AppConfig (thresholds, paths, …).
    store:
        Pre-loaded FaceVectorStore.
    """

    def __init__(self, config: AppConfig, store: FaceVectorStore) -> None:
        self._cfg = config
        self._store = store
        self._id_cfg = config.identification

        # Initialise pipeline components
        self._det = FaceDetector(config.model.detector_path, config.detection)
        self._aln = FaceAligner(output_size=112)
        self._emb = FaceEmbedder(config.model.embedder_path, self._id_cfg.embedding_dim)
        self._qlt = QualityChecker(config.quality)
        self._lv  = LivenessChecker()                # stub

        # FPS ring buffer (last 30 frame timestamps)
        self._ts_buf: deque[float] = deque(maxlen=30)
        self._last_latency_ms: float = 0.0

        config.debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: AppConfig = CONFIG) -> "WebcamDemo":
        """Build from global config; loads FAISS index from disk."""
        store = FaceVectorStore(config.identification.embedding_dim)
        store.load(config.data.faiss_index_path, config.data.metadata_path)
        logger.info(
            f"FAISS index loaded: {store.total_vectors} vectors, "
            f"{len(store.enrolled_employees)} employees enrolled."
        )
        return cls(config, store)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open webcam and run until user presses 'q'."""
        cap = cv2.VideoCapture(self._cfg.webcam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam index={self._cfg.webcam_index}. "
                "Check that a camera is connected."
            )
        logger.info("Webcam opened. Press 'q' to quit, 's' to save a debug frame.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame — retrying…")
                    time.sleep(0.05)
                    continue

                frame = self._process_frame(frame)
                cv2.imshow("NFD Face Attendance", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    self._save_debug(frame)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam closed.")

    # ── Frame processing ───────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run full detection + identification pipeline on one frame."""
        with Timer() as t:
            detections = self._det.detect(frame)

        self._last_latency_ms = t.elapsed_ms
        self._ts_buf.append(time.perf_counter())
        fps = self._compute_fps()

        # ── No faces ──────────────────────────────────────────────────────────
        if not detections:
            draw_hud(frame, fps, t.elapsed_ms)
            draw_status(frame, "No face detected", (100, 100, 100))
            return frame

        # ── Multiple faces — warn but still process each ───────────────────────
        if len(detections) > 1:
            draw_status(frame, f"Multiple faces ({len(detections)}) — step closer", (30, 160, 255))

        # ── Per-face inference ─────────────────────────────────────────────────
        for det in detections:
            result = self._identify_face(frame, det)

            # Log to console
            if self._cfg.log_identifications:
                log_identification(result, logger)

            draw_face(frame, det, result)

        draw_hud(frame, fps, t.elapsed_ms)
        return frame

    def _identify_face(
        self,
        frame: np.ndarray,
        det: FaceDetection,
    ) -> IdentificationResult:
        """Run quality → liveness → align → embed → search → decide."""
        with Timer() as t:
            # 1. Quality gate
            quality = self._qlt.check(frame, det)
            if not quality.ok:
                return self._make_result(
                    "LOW_QUALITY", None, None, 0.0, 0.0,
                    quality, LivenessResult(True, 1.0, "stub"), t.elapsed_ms,
                )

            # 2. Align
            aligned = self._aln.align(frame, det)

            # 3. Liveness
            liveness = self._lv.check(frame, det, aligned)
            if not liveness.is_live:
                return self._make_result(
                    "SPOOF_SUSPECTED", None, None, 0.0, 0.0,
                    quality, liveness, t.elapsed_ms,
                )

            # 4. Embed
            embedding = self._emb.embed(aligned)

            # 5. FAISS search
            candidates: list[SearchResult] = self._store.search(
                embedding, top_k=self._id_cfg.top_k
            )

        return self._decide(candidates, quality, liveness, t.elapsed_ms)

    # ── Decision logic ─────────────────────────────────────────────────────────

    def _decide(
        self,
        candidates: list[SearchResult],
        quality: QualityResult,
        liveness: LivenessResult,
        latency_ms: float,
    ) -> IdentificationResult:
        """
        Apply ACCEPT / UNKNOWN thresholds.

        ACCEPT requires:
          quality_ok AND liveness_ok
          AND top1_score >= ACCEPT_THRESHOLD
          AND top1_score - top2_score >= MARGIN_THRESHOLD

        The margin check guards against ambiguous cases where two employees
        have similar scores (would be a low-confidence identification).
        """
        if not candidates:
            return self._make_result(
                "UNKNOWN", None, None, 0.0, 0.0, quality, liveness, latency_ms
            )

        top1 = candidates[0]
        top2_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top1.score - top2_score

        accept = (
            top1.score >= self._id_cfg.accept_threshold
            and margin >= self._id_cfg.margin_threshold
        )

        decision = "ACCEPT" if accept else "UNKNOWN"
        emp_id = top1.employee_id if accept else None
        name   = top1.name        if accept else None

        return self._make_result(
            decision, emp_id, name, top1.score, margin,
            quality, liveness, latency_ms,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_result(
        decision: str,
        employee_id, name,
        score: float, margin: float,
        quality: QualityResult,
        liveness: LivenessResult,
        latency_ms: float,
    ) -> IdentificationResult:
        return IdentificationResult(
            decision=decision,
            employee_id=employee_id,
            name=name,
            score=score,
            margin=margin,
            quality=quality,
            liveness=liveness,
            latency_ms=latency_ms,
        )

    def _compute_fps(self) -> float:
        if len(self._ts_buf) < 2:
            return 0.0
        elapsed = self._ts_buf[-1] - self._ts_buf[0]
        return (len(self._ts_buf) - 1) / elapsed if elapsed > 0 else 0.0

    def _save_debug(self, frame: np.ndarray) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._cfg.debug_dir / f"frame_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info(f"Debug frame saved → {path}")
