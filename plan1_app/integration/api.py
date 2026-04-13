"""Minimal FastAPI service (optional JPEG frame in body for demo)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
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


app = FastAPI(title="Plan1 Attendance", version="0.1.0")


class Health(BaseModel):
    status: str


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok")


@app.post("/recognize_upload")
async def recognize_upload(file: UploadFile = File(...), store_frame: bool = False):
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="invalid image")
    rec = get_recognizer()
    out = rec.recognize_frame(bgr)
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
    # do not return raw embedding by default
    out.pop("embedding", None)
    return out
