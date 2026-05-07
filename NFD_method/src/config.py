"""
Centralized configuration for NFD (Neural Face Detection) attendance system.

All thresholds, paths, and hyperparameters are defined here.
Edit this file to tune the pipeline — do NOT hard-code values elsewhere.

Tuning guide (IdentificationConfig):
  - Too many false accepts (wrong person accepted):
      Increase ACCEPT_THRESHOLD or MARGIN_THRESHOLD.
  - Too many UNKNOWN rejections (right person not recognized):
      Decrease ACCEPT_THRESHOLD slightly, OR add more enrollment images.
  - In low-light environments:
      Improve camera / lighting before touching thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = NFD_method/
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    """Filesystem paths to ONNX model files."""
    detector_path: Path = field(default_factory=lambda: ROOT / "models" / "scrfd.onnx")
    embedder_path: Path = field(default_factory=lambda: ROOT / "models" / "arcface.onnx")


@dataclass
class DataConfig:
    """Paths to data directories and FAISS index files."""
    employees_dir: Path = field(default_factory=lambda: ROOT / "data" / "employees")
    embeddings_dir: Path = field(default_factory=lambda: ROOT / "data" / "embeddings")

    @property
    def faiss_index_path(self) -> Path:
        return self.embeddings_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.embeddings_dir / "metadata.json"


@dataclass
class DetectionConfig:
    """SCRFD detector parameters."""
    input_size: tuple[int, int] = (640, 640)
    conf_threshold: float = 0.5    # detector score cutoff
    nms_threshold: float = 0.4     # non-max suppression IoU cutoff
    max_faces: int = 10            # cap per frame


@dataclass
class QualityConfig:
    """
    Face quality gate parameters.
    Tune based on your camera and lighting conditions.
    """
    # Bounding-box size
    min_face_size: int = 80             # px — minimum width AND height

    # Sharpness: variance of Laplacian
    blur_threshold: float = 60.0       # below this → "blur" rejection

    # Exposure: grayscale mean [0-255]
    min_brightness: float = 40.0       # too dark
    max_brightness: float = 220.0      # overexposed

    # Face must not clip the frame edge (ratio of frame dimension)
    edge_margin_ratio: float = 0.05

    # Landmark sanity: inter-eye distance in pixels
    min_eye_distance: float = 20.0


@dataclass
class IdentificationConfig:
    """
    Decision thresholds for 1:N face identification.

    Cosine similarity range: 0 (orthogonal / unrelated) → 1 (identical).
    Typical ArcFace scores for the same person: 0.40 – 0.85.
    """
    accept_threshold: float = 0.45    # minimum top-1 score to ACCEPT
    margin_threshold: float = 0.05    # minimum gap between top-1 and top-2 employee
    top_k: int = 5                    # number of FAISS candidates to retrieve
    embedding_dim: int = 512          # ArcFace output dimension


@dataclass
class AppConfig:
    """Root configuration — passed to all subsystems."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)

    # Webcam device index (0 = first camera)
    webcam_index: int = 0

    # Directory for saved debug frames (press 's' in webcam window)
    debug_dir: Path = field(default_factory=lambda: ROOT / "debug")

    # Print an identification log line to console on every decision
    log_identifications: bool = True


# ── Global singleton — import this everywhere ─────────────────────────────────
CONFIG = AppConfig()
