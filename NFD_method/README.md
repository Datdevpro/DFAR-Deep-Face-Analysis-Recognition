# NFD — Neural Face Detection Attendance System

Local PC face-recognition attendance prototype.

**Pipeline:** SCRFD (detect) → ArcFace ONNX (512-D embed) → FAISS IndexFlatIP (search) → Decision logic

---

## Quick Start

### 1. Install dependencies

```bash
cd NFD_method
pip install -r requirements.txt
```

> **GPU note:** For ONNX on NVIDIA GPU replace `onnxruntime` with `onnxruntime-gpu`.

---

### 2. Download ONNX models

Place the following files in `models/`:

| File | Source |
|---|---|
| `models/scrfd.onnx` | [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) — recommended: `scrfd_10g_bnkps.onnx` |
| `models/arcface.onnx` | [InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) — recommended: `buffalo_l` / `w600k_r50.onnx` |

---

### 3. Prepare employee images

```
data/employees/
  EMP001/
    1.jpg
    2.jpg
    3.jpg
  EMP002/
    1.jpg
```

- **1–5 images per person** recommended.
- Images should be clear, well-lit, front-facing.
- Each folder name becomes the `employee_id`.
- Optionally add `name.txt` (plain text, one line) for the display name.

---

### 4. Enroll employees

```bash
python scripts/enroll_employee.py \
    --employee-id EMP001 \
    --name "Nguyen Van A" \
    --image-dir data/employees/EMP001
```

Enrollment is **incremental** — re-running adds new vectors without deleting existing ones.

To rebuild the entire index from scratch:

```bash
python scripts/build_index.py
```

---

### 5. Run webcam demo

```bash
python scripts/run_webcam.py
```

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save debug frame to `debug/` |

#### On-screen display

| Overlay | Meaning |
|---|---|
| Green box + name | **ACCEPT** — identity confirmed |
| Orange box + UNKNOWN | Score below threshold |
| Blue box + LOW_QUALITY | Quality gate failed |
| FPS / Latency | Top-left HUD |
| Q: score | Quality score per face |

---

### 6. Test components individually

```bash
# Test detector, embedder, FAISS (with a known test image)
python scripts/test_pipeline.py --image data/employees/EMP001/1.jpg

# Skip webcam test
python scripts/test_pipeline.py --image data/employees/EMP001/1.jpg --skip-webcam
```

---

## Threshold Tuning (`src/config.py`)

```python
accept_threshold: float = 0.45   # minimum cosine similarity to ACCEPT
margin_threshold: float = 0.05   # minimum gap between top-1 and top-2 employee
```

| Symptom | Action |
|---|---|
| Wrong person accepted | **Increase** `ACCEPT_THRESHOLD` or `MARGIN_THRESHOLD` |
| Right person always UNKNOWN | **Decrease** `ACCEPT_THRESHOLD` slightly, or **add more enrollment images** |
| Blurry frame rejected | **Decrease** `blur_threshold` in `QualityConfig`, or improve lighting |
| Face too small rejected | Move closer to camera, or **decrease** `min_face_size` |

Typical ArcFace cosine similarity ranges:
- Same person: **0.40 – 0.85**
- Different persons: **−0.1 – 0.35**

---

## Project Structure

```
NFD_method/
├── models/
│   ├── scrfd.onnx          ← face detector (download separately)
│   └── arcface.onnx        ← face embedder (download separately)
├── data/
│   ├── employees/          ← raw enrollment images per employee
│   └── embeddings/
│       ├── faiss.index     ← auto-generated after enrollment
│       └── metadata.json   ← vector_id → employee mapping
├── src/
│   ├── config.py           ← all thresholds & paths (edit here to tune)
│   ├── detector.py         ← SCRFD ONNX face detector
│   ├── aligner.py          ← 5-point affine alignment → 112×112
│   ├── embedder.py         ← ArcFace ONNX → 512-D L2-normalised embedding
│   ├── quality.py          ← blur / brightness / size / landmark checks
│   ├── liveness.py         ← stub anti-spoofing (interface for future model)
│   ├── vector_store.py     ← FAISS IndexFlatIP store + metadata
│   ├── enrollment.py       ← enrollment pipeline
│   ├── webcam_demo.py      ← real-time webcam loop + decision logic
│   └── utils.py            ← dataclasses, drawing, logger, Timer
├── scripts/
│   ├── enroll_employee.py  ← enroll one employee from CLI
│   ├── build_index.py      ← rebuild entire index from scratch
│   ├── run_webcam.py       ← launch webcam demo
│   └── test_pipeline.py    ← quick component tests
├── debug/                  ← saved debug frames (press 's' in webcam)
├── requirements.txt
└── README.md
```

---

## Liveness (Anti-Spoofing)

Currently a **stub** — always returns `is_live=True`.

To integrate a real model (e.g., MiniFASNet, Silent-Face):
1. Place ONNX model in `models/`.
2. Edit `src/liveness.py` → load session in `__init__`, implement `_stub_check()`.
3. No other files need to change.

---

## FAQ

**Q: What SCRFD variant should I use?**  
A: `scrfd_10g_bnkps.onnx` gives the best accuracy with landmarks. `scrfd_2.5g_bnkps.onnx` is faster for edge devices.

**Q: My enrollment images all get rejected for quality.**  
A: Check `blur_threshold` and `min_brightness` in `src/config.py`. Lower them temporarily, or improve image quality.

**Q: FAISS index is empty but enrollment completed.**  
A: Run `test_pipeline.py` to verify the detector and embedder are working correctly.
