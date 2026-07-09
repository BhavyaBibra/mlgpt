"""Drift detection: compares recent prediction windows against reference data.

Uses PSI (Population Stability Index) computed directly for numeric features —
simple, explainable, and dependency-light for Phase 1. Evidently suites slot in
here in Phase 2 for richer checks (JS distance, data quality, prediction drift).
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (
    DRIFT_WINDOW_MINUTES, MIN_ROWS_FOR_DRIFT, PSI_THRESHOLD, SessionLocal,
)
from app.models.db import DriftReport, Incident, Prediction

REFERENCE_DIR = os.getenv("REFERENCE_DIR", "/tmp/mlgpt_reference")
os.makedirs(REFERENCE_DIR, exist_ok=True)


# ---------- reference data ----------

def save_reference(model_id: str, df: pd.DataFrame) -> None:
    df.to_parquet(os.path.join(REFERENCE_DIR, f"{model_id}.parquet"))


def load_reference(model_id: str) -> pd.DataFrame | None:
    path = os.path.join(REFERENCE_DIR, f"{model_id}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


# ---------- PSI ----------

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------- the scheduled check ----------

def run_drift_check() -> None:
    db: Session = SessionLocal()
    try:
        model_ids = [r[0] for r in db.execute(select(Prediction.model_id).distinct())]
        for model_id in model_ids:
            _check_model(db, model_id)
    finally:
        db.close()


def _check_model(db: Session, model_id: str) -> None:
    reference = load_reference(model_id)
    if reference is None:
        return

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=DRIFT_WINDOW_MINUTES)
    rows = db.execute(
        select(Prediction.features, Prediction.prediction)
        .where(Prediction.model_id == model_id, Prediction.ts >= window_start)
    ).all()
    if len(rows) < MIN_ROWS_FOR_DRIFT:
        return

    current = pd.DataFrame([dict(r.features, __prediction__=r.prediction) for r in rows])
    flagged_features: list[tuple[str, float]] = []

    numeric_cols = [
        c for c in reference.columns
        if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]
    for col in numeric_cols + ["__prediction__"] if "__prediction__" in reference.columns else numeric_cols:
        if col not in current.columns:
            continue
        value = psi(
            reference[col].to_numpy(dtype=float),
            pd.to_numeric(current[col], errors="coerce").to_numpy(dtype=float),
        )
        flagged = value >= PSI_THRESHOLD
        db.add(DriftReport(
            model_id=model_id, window_start=window_start, window_end=now,
            feature=col, metric="psi", value=round(value, 4),
            threshold=PSI_THRESHOLD, flagged=flagged,
        ))
        if flagged:
            flagged_features.append((col, value))

    db.commit()

    if flagged_features:
        _open_incident_if_needed(db, model_id, flagged_features)


def _open_incident_if_needed(db: Session, model_id: str, flagged: list[tuple[str, float]]) -> None:
    """One open incident per model at a time (dedup/cooldown)."""
    existing = db.execute(
        select(Incident).where(Incident.model_id == model_id, Incident.status == "open")
    ).scalar_one_or_none()
    if existing:
        return
    worst = max(flagged, key=lambda t: t[1])
    summary = (
        f"Drift detected in {len(flagged)} feature(s); worst: "
        f"{worst[0]} (PSI {worst[1]:.2f}, threshold {PSI_THRESHOLD})"
    )
    db.add(Incident(
        model_id=model_id,
        severity="critical" if worst[1] >= 2 * PSI_THRESHOLD else "warning",
        trigger_summary=summary,
    ))
    db.commit()
