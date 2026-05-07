"""SQLite persistence for attendance events."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, LargeBinary, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class AttendanceLog(Base):
    __tablename__ = "attendance_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    employee_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    det_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    frame_jpeg: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


def make_session(db_path: str | Path):
    engine = create_engine(f"sqlite:///{Path(db_path).resolve()}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False, future=True)


def log_match(
    session_factory,
    *,
    employee_id: str | None,
    margin: float | None,
    det_prob: float | None,
    note: str | None,
    frame_jpeg: bytes | None = None,
) -> None:
    with session_factory() as s:
        s.add(
            AttendanceLog(
                employee_id=employee_id,
                margin=margin,
                det_prob=det_prob,
                note=note,
                frame_jpeg=frame_jpeg,
            )
        )
        s.commit()
