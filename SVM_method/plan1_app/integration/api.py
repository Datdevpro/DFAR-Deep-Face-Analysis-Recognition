"""FastAPI service for Plan1 face recognition."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from plan1_app.integration.db import log_match, make_session

# Lazy recognizer singleton for embedded / low-RAM hosts
_recognizer = None
_session_factory = None


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from plan1_app.pipeline.recognize import Plan1Recognizer

        art = os.environ.get("PLAN1_SVM_PATH", str(Path(__file__).resolve().parents[2] / "artifacts" / "svm_plan1.joblib"))
        if not Path(art).is_file():
            raise RuntimeError(f"Missing SVM artifact: {art} (set PLAN1_SVM_PATH)")
        dev = os.environ.get("PLAN1_DEVICE")
        _recognizer = Plan1Recognizer(svm_artifact=art, device=dev)
    return _recognizer


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        dbp = os.environ.get("PLAN1_DB_PATH", str(Path(__file__).resolve().parents[2] / "attendance.sqlite3"))
        _session_factory = make_session(dbp)
    return _session_factory


app = FastAPI(
    title="DFAR Face Recognition API",
    version="1.0.0",
    description=(
        "REST API for image-based face recognition using "
        "MTCNN + dlib embeddings + Linear SVM."
    ),
)


class Health(BaseModel):
    status: str


class RecognitionResponse(BaseModel):
    ok: bool
    reason: str
    employee_id: str | None
    margin: float | None
    det_prob: float | None
    box: list[float] | None
    note: str | None
    model_id: str
    latency_ms: int


def _recognize_from_upload(file_bytes: bytes, store_frame: bool = False) -> RecognitionResponse:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="invalid image")

    start = time.perf_counter()
    rec = get_recognizer()
    out: dict[str, Any] = rec.recognize_frame(bgr)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if out.get("ok") and store_frame:
        ok, buf = cv2.imencode(".jpg", bgr)
        jpeg = buf.tobytes() if ok else None
        log_match(
            get_session_factory(),
            employee_id=out.get("employee_id"),
            margin=out.get("margin"),
            det_prob=out.get("det_prob"),
            note=out.get("note"),
            frame_jpeg=jpeg,
        )

    # Keep API payload compact and JSON-safe.
    out.pop("embedding", None)
    out["model_id"] = Path(os.environ.get("PLAN1_SVM_PATH", "artifacts/svm_plan1.joblib")).name
    out["latency_ms"] = elapsed_ms
    return RecognitionResponse(**out)


@app.get("/health", response_model=Health, tags=["legacy"])
def health():
    return Health(status="ok")


@app.get("/v1/health", response_model=Health, tags=["system"])
def health_v1():
    return Health(status="ok")


@app.get("/v1/ready", response_model=Health, tags=["system"])
def ready_v1():
    # Ready means model artifact exists and recognizer can be initialized.
    _ = get_recognizer()
    return Health(status="ready")


@app.post(
    "/v1/recognition/image",
    response_model=RecognitionResponse,
    tags=["recognition"],
    summary="Test one image against active model",
)
async def recognize_image_v1(
    file: UploadFile = File(..., description="Input image file (jpg/png/jpeg)."),
    store_frame: bool = Form(False, description="If true, store successful match frame to DB."),
):
    data = await file.read()
    return _recognize_from_upload(data, store_frame=store_frame)


@app.post("/recognize_upload", response_model=RecognitionResponse, tags=["legacy"])
async def recognize_upload(file: UploadFile = File(...), store_frame: bool = False):
    """Backward-compatible endpoint; prefer `/v1/recognition/image`."""
    data = await file.read()
    return _recognize_from_upload(data, store_frame=store_frame)
