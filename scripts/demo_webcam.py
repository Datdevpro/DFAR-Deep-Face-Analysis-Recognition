"""Live demo: webcam → Plan1Recognizer (press q to quit)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plan1_app.pipeline.recognize import Plan1Recognizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "svm_plan1.joblib")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    rec = Plan1Recognizer(svm_artifact=args.artifact, device=args.device)
    cap = cv2.VideoCapture(args.camera)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = rec.recognize_frame(frame)
        text = out.get("employee_id") or out.get("reason", "?")
        cv2.putText(frame, str(text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if out.get("box") is not None:
            x1, y1, x2, y2 = [int(v) for v in out["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("plan1", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
