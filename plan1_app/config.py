"""Paths and defaults for Plan 1."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

SHAPE_PREDICTOR = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

# MTCNN → dlib alignment crop
CROP_SIZE = 150
EMBED_DIM = 128
