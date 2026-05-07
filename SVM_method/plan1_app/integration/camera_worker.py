"""Background capture thread → optional callback (recognition + logging)."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2


class CameraWorker(threading.Thread):
    """
    Reads frames from an IP camera or device index in a daemon thread.
    ``frame_callback`` receives BGR ``numpy`` frames; keep work light or offload.
    """

    def __init__(
        self,
        source: str | int,
        frame_callback: Callable[[Any], None],
        queue_size: int = 2,
        reconnect_sec: float = 3.0,
    ) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.frame_callback = frame_callback
        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.reconnect_sec = reconnect_sec

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cap: cv2.VideoCapture | None = None
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    time.sleep(self.reconnect_sec)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                time.sleep(self.reconnect_sec)
                continue
            try:
                self.frame_callback(frame)
            except Exception:
                pass

        if cap is not None:
            cap.release()
