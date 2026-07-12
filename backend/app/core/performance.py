"""Model performance over time: joins ground-truth `actuals` back to the
`predictions` that produced them and computes accuracy / ROC-AUC per window.

This is what turns "a feature drifted" into "and here is the accuracy it cost" —
the signal the root-cause agent needs to tell a harmless input shift from a real
degradation, and to distinguish a deploy bug (AUC collapses) from a population
shift (AUC holds, accuracy/calibration erodes).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db import Actual, Prediction


def _rows(db: Session, model_id: str, start: datetime | None, end: datetime | None):
    q = (
        select(Prediction.ts, Prediction.prediction, Actual.actual)
        .join(Actual, Actual.prediction_id == Prediction.id)
        .where(Prediction.model_id == model_id)
    )
    if start is not None:
        q = q.where(Prediction.ts >= start)
    if end is not None:
        q = q.where(Prediction.ts < end)
    return db.execute(q).all()


def _metrics(scores: list[float], actuals: list[float], threshold: float) -> dict:
    n = len(actuals)
    labels = [1 if a >= 0.5 else 0 for a in actuals]
    preds = [1 if s >= threshold else 0 for s in scores]
    accuracy = sum(p == y for p, y in zip(preds, labels)) / n
    default_rate = sum(labels) / n

    auc = None
    if 0 < sum(labels) < n:  # AUC undefined for a single-class window
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, scores))

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "default_rate": round(default_rate, 4),
    }


def compute_window(db: Session, model_id: str,
                   start: datetime | None = None, end: datetime | None = None,
                   threshold: float = 0.5) -> dict | None:
    """Accuracy/AUC over one window; None if there's no labelled traffic yet."""
    rows = _rows(db, model_id, start, end)
    if not rows:
        return None
    scores = [float(r.prediction) for r in rows]
    actuals = [float(r.actual) for r in rows]
    out = _metrics(scores, actuals, threshold)
    out["window_start"] = start
    out["window_end"] = end
    return out


def performance_timeseries(db: Session, model_id: str, buckets: int = 12,
                           threshold: float = 0.5) -> list[dict]:
    """Bucket all labelled predictions across their time span into `buckets`
    equal windows and compute metrics per bucket — the accuracy timeline."""
    rows = _rows(db, model_id, None, None)
    if not rows:
        return []
    times = [r.ts for r in rows]
    t0, t1 = min(times), max(times)
    span = (t1 - t0).total_seconds()
    if span <= 0:
        one = _metrics([float(r.prediction) for r in rows],
                       [float(r.actual) for r in rows], threshold)
        one["window_start"], one["window_end"] = t0, t1
        return [one]

    step = span / buckets
    out = []
    for b in range(buckets):
        lo = t0 + timedelta(seconds=b * step)
        hi = t0 + timedelta(seconds=(b + 1) * step)
        sel = [r for r in rows if (r.ts >= lo and (r.ts < hi or b == buckets - 1))]
        if not sel:
            continue
        m = _metrics([float(r.prediction) for r in sel],
                     [float(r.actual) for r in sel], threshold)
        m["window_start"], m["window_end"] = lo, hi
        out.append(m)
    return out
