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
    DRIFT_WINDOW_MINUTES, MIN_ROWS_FOR_DRIFT, NULL_RATE_THRESHOLD,
    PERF_ACC_DROP, PERF_AUC_DROP, PERF_MIN_LABELED, PSI_THRESHOLD, SessionLocal,
)
from app.core.performance import compute_window
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
        cur_col = pd.to_numeric(current[col], errors="coerce")
        value = psi(
            reference[col].to_numpy(dtype=float),
            cur_col.to_numpy(dtype=float),
        )
        flagged = value >= PSI_THRESHOLD
        db.add(DriftReport(
            model_id=model_id, window_start=window_start, window_end=now,
            feature=col, metric="psi", value=round(value, 4),
            threshold=PSI_THRESHOLD, flagged=flagged,
        ))
        if flagged:
            flagged_features.append((col, value))

        # null-rate check: PSI ignores NaN, so a null spike is invisible to it.
        # A feature that suddenly goes missing is a classic data-quality break.
        if col != "__prediction__":
            ref_null = float(pd.to_numeric(reference[col], errors="coerce").isna().mean())
            cur_null = float(cur_col.isna().mean())
            null_flagged = cur_null >= max(NULL_RATE_THRESHOLD, ref_null + 0.2)
            db.add(DriftReport(
                model_id=model_id, window_start=window_start, window_end=now,
                feature=col, metric="null_rate", value=round(cur_null, 4),
                threshold=NULL_RATE_THRESHOLD, flagged=null_flagged,
            ))
            if null_flagged:
                flagged_features.append((f"{col} (null spike)", cur_null))

    db.commit()

    perf_decay = _performance_decayed(db, model_id, window_start, now)
    if perf_decay:
        # record the decay as a citable evidence row — for concept/world drift
        # this is the ONLY real signal, so the agent must be able to cite it
        db.add(DriftReport(
            model_id=model_id, window_start=window_start, window_end=now,
            feature="__performance__", metric="perf_decay",
            value=perf_decay["recent_acc"], threshold=perf_decay["baseline_acc"],
            flagged=True,
        ))
        db.commit()
    if flagged_features or perf_decay:
        _open_incident_if_needed(db, model_id, flagged_features, perf_decay)


def _performance_decayed(db: Session, model_id: str, window_start, now) -> dict | None:
    """Recent labelled window vs the earlier baseline. Returns the drop if
    accuracy or AUC fell past threshold — the only signal that catches concept
    drift, which leaves inputs looking normal."""
    recent = compute_window(db, model_id, start=window_start, end=now)
    baseline = compute_window(db, model_id, end=window_start)
    if not recent or not baseline or recent["n"] < PERF_MIN_LABELED:
        return None
    acc_drop = baseline["accuracy"] - recent["accuracy"]
    auc_drop = (baseline["auc"] - recent["auc"]) \
        if baseline["auc"] is not None and recent["auc"] is not None else 0.0
    if acc_drop >= PERF_ACC_DROP or auc_drop >= PERF_AUC_DROP:
        return {"acc_drop": round(acc_drop, 4), "auc_drop": round(auc_drop, 4),
                "recent_acc": recent["accuracy"], "baseline_acc": baseline["accuracy"]}
    return None


def _open_incident_if_needed(db: Session, model_id: str,
                             flagged: list[tuple[str, float]],
                             perf_decay: dict | None = None) -> None:
    """One open incident per model at a time (dedup/cooldown)."""
    existing = db.execute(
        select(Incident).where(Incident.model_id == model_id, Incident.status == "open")
    ).scalar_one_or_none()
    if existing:
        return
    if flagged:
        worst = max(flagged, key=lambda t: t[1])
        summary = (f"Drift detected in {len(flagged)} feature(s); worst: "
                   f"{worst[0]} ({worst[1]:.2f})")
        severity = "critical" if worst[1] >= 2 * PSI_THRESHOLD else "warning"
    else:  # performance-only trigger (concept/world drift)
        summary = (f"Performance decay: accuracy {perf_decay['baseline_acc']:.2f} "
                   f"-> {perf_decay['recent_acc']:.2f} with no input drift flagged")
        severity = "critical" if perf_decay["acc_drop"] >= 2 * PERF_ACC_DROP else "warning"
    db.add(Incident(model_id=model_id, severity=severity, trigger_summary=summary))
    db.commit()
