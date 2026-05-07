"""Cross-check face attendance logs against RFID swipe logs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class RfidEvent:
    employee_id: str
    ts: datetime


def load_rfid_csv(path: str | Path) -> list[RfidEvent]:
    """Expect columns: employee_id, timestamp (ISO8601)."""
    rows: list[RfidEvent] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                RfidEvent(
                    employee_id=str(row["employee_id"]).strip(),
                    ts=datetime.fromisoformat(row["timestamp"]),
                )
            )
    return rows


def match_within_window(
    face_ts: datetime,
    employee_id: str,
    rfid_events: list[RfidEvent],
    window: timedelta,
) -> bool:
    for ev in rfid_events:
        if ev.employee_id != employee_id:
            continue
        if abs((face_ts - ev.ts).total_seconds()) <= window.total_seconds():
            return True
    return False
