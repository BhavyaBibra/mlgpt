"""MLGPT database schema.

Six tables: predictions, actuals, events, drift_reports, incidents, explanations.
Everything the root-cause agent will later reason over lives here.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    features: Mapped[dict] = mapped_column(JSON)
    prediction: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), default="unknown")

    __table_args__ = (Index("ix_pred_model_ts", "model_id", "ts"),)


class Actual(Base):
    __tablename__ = "actuals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    actual: Mapped[float] = mapped_column(Float)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="note")  # deploy | config | pipeline | note
    description: Mapped[str] = mapped_column(Text)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    feature: Mapped[str] = mapped_column(String(128))
    metric: Mapped[str] = mapped_column(String(32))       # psi | js_distance | pred_drift
    value: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    opened_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    severity: Mapped[str] = mapped_column(String(16), default="warning")  # warning | critical
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open | resolved
    trigger_summary: Mapped[str] = mapped_column(Text, default="")


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    summary: Mapped[str] = mapped_column(Text)
    root_cause: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list] = mapped_column(JSON, default=list)   # [{type, ref_id, detail}]
    confidence: Mapped[str] = mapped_column(String(16), default="medium")
    suggested_action: Mapped[str] = mapped_column(Text, default="")
