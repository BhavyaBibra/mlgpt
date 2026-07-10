"""Incident context assembler — the evidence packet the root-cause agent reasons over.

This is deliberately NOT a raw data dump. Its job is to surface the signals that
DISCRIMINATE between the ways a model fails, because naming the cause correctly
is the whole product:

  - deploy / pipeline break : a feature's PSI explodes AND ranking (AUC) collapses,
                              usually right after a deploy event.
  - data-quality break      : same, but the shift is a null spike or a value
                              collapse (std->0, one value dominates) — a pipeline
                              subtype.
  - world / population change: little or no single-feature input drift, AUC roughly
                              holds, but accuracy/calibration erodes; no deploy
                              correlates.

Every number here traces to a real row (drift_report / event / prediction) so the
agent can cite it and we can enforce citations. `classify_signature` computes a
rules-based prior + confidence that is testable on its own and forms the backbone
of the labeled scenario suite.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.drift import load_reference
from app.core.performance import compute_window
from app.models.db import DriftReport, Event, Incident, Prediction

EVENT_WINDOW_HOURS = 48
TOP_FEATURES = 6


def _naive(dt: datetime | None) -> datetime | None:
    """Normalize to naive UTC so Postgres (aware) and SQLite (naive) compare."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _drift_snapshot(db: Session, model_id: str, opened_ts: datetime):
    """The batch of drift reports that opened the incident: the latest window at
    or just before the incident time."""
    latest_end = db.execute(
        select(DriftReport.window_end)
        .where(DriftReport.model_id == model_id,
               DriftReport.window_end <= opened_ts + timedelta(minutes=2))
        .order_by(desc(DriftReport.window_end)).limit(1)
    ).scalar_one_or_none()
    if latest_end is None:
        return None, []
    reports = db.execute(
        select(DriftReport).where(DriftReport.model_id == model_id,
                                  DriftReport.window_end == latest_end)
    ).scalars().all()
    return latest_end, reports


def _feature_shifts(model_id: str, features: list[str], onset, window_end,
                    db: Session) -> list[dict]:
    """Pre (reference) vs post (drift window) stats for the top drifted features —
    the evidence for HOW a feature moved (mean jump, null spike, value collapse)."""
    reference = load_reference(model_id)
    rows = db.execute(
        select(Prediction.features).where(
            Prediction.model_id == model_id,
            Prediction.ts >= onset, Prediction.ts <= window_end)
    ).all()
    current = pd.DataFrame([r.features for r in rows]) if rows else pd.DataFrame()

    out = []
    for f in features:
        item = {"feature": f}
        if reference is not None and f in reference.columns:
            ref = pd.to_numeric(reference[f], errors="coerce").dropna()
            item["pre_mean"] = round(float(ref.mean()), 3)
            item["pre_std"] = round(float(ref.std()), 3)
        if not current.empty and f in current.columns:
            cur = pd.to_numeric(current[f], errors="coerce")
            item["post_null_rate"] = round(float(cur.isna().mean()), 3)
            cur = cur.dropna()
            if len(cur):
                item["post_mean"] = round(float(cur.mean()), 3)
                item["post_std"] = round(float(cur.std()), 3)
                item["post_mode_share"] = round(float(cur.value_counts(normalize=True).iloc[0]), 3)
        out.append(item)
    return out


def classify_signature(signals: dict) -> dict:
    """Rules-based prior over {deploy_or_pipeline, data_quality, world_change,
    ambiguous} with a confidence and a cited rationale. Testable on its own."""
    max_psi = signals["max_psi"]
    auc_delta = signals["auc_delta"] or 0.0
    acc_delta = signals["accuracy_delta"] or 0.0
    deploy = signals["nearest_deploy_hours"]  # signed hrs from onset, or None
    shifts = signals["feature_shifts"]

    deploy_correlates = deploy is not None and -EVENT_WINDOW_HOURS <= deploy <= 6
    big_input_drift = max_psi >= 1.0
    auc_collapsed = auc_delta >= 0.03
    auc_held = auc_delta < 0.02
    quality_flag = any(
        (s.get("post_null_rate", 0) >= 0.3) or (s.get("post_mode_share", 0) >= 0.9)
        for s in shifts
    )

    if big_input_drift and (auc_collapsed or quality_flag):
        label = "data_quality" if quality_flag else "deploy_or_pipeline"
        if deploy_correlates:
            conf, why = "high", (
                f"A feature's distribution broke (PSI {max_psi:.1f}) and AUC fell "
                f"{auc_delta:.3f}, {abs(deploy):.1f}h "
                f"{'after' if deploy <= 0 else 'before'} a deploy event"
                + (" with a null spike / value collapse" if quality_flag else "")
                + " — a shipped change, not a changed world."
            )
        else:
            conf, why = "medium", (
                f"Feature broke (PSI {max_psi:.1f}) and AUC fell {auc_delta:.3f}, "
                f"but no deploy event correlates — likely a pipeline/data-quality "
                f"break; check upstream sources."
            )
    elif not big_input_drift and auc_held and acc_delta >= 0.03:
        label, conf, why = "world_change", "medium", (
            f"Accuracy fell {acc_delta:.3f} while AUC held ({auc_delta:+.3f}) with "
            f"no single-feature input drift (max PSI {max_psi:.2f}) and no deploy "
            f"nearby — the population/relationship changed, not the pipeline."
        )
    else:
        label, conf, why = "ambiguous", "low", (
            f"Signals are mixed (max PSI {max_psi:.2f}, AUC Δ {auc_delta:.3f}, "
            f"accuracy Δ {acc_delta:.3f}) — needs a human look."
        )
    return {"label": label, "confidence": conf, "rationale": why}


def assemble_incident_context(db: Session, incident_id: int) -> dict | None:
    """Build the full, citation-ready evidence packet for one incident."""
    incident = db.get(Incident, incident_id)
    if incident is None:
        return None
    model_id = incident.model_id
    opened = _naive(incident.opened_ts)

    window_end, reports = _drift_snapshot(db, model_id, opened)
    window_end = _naive(window_end)
    flagged = sorted([r for r in reports if r.flagged and r.feature != "__prediction__"],
                     key=lambda r: r.value, reverse=True)
    onset = _naive(reports[0].window_start) if reports else opened
    pred_drift = next((r.value for r in reports if r.feature == "__prediction__"), None)

    # events within +/-48h of drift onset, with signed hours (negative = before)
    ev_lo, ev_hi = onset - timedelta(hours=EVENT_WINDOW_HOURS), onset + timedelta(hours=EVENT_WINDOW_HOURS)
    events = db.execute(
        select(Event).where(Event.model_id == model_id,
                            Event.ts >= ev_lo, Event.ts <= ev_hi)
        .order_by(Event.ts)
    ).scalars().all()
    event_list = [{
        "event_id": e.id, "kind": e.kind, "description": e.description,
        "ts": _naive(e.ts),
        "hours_from_onset": round((_naive(e.ts) - onset).total_seconds() / 3600, 2),
    } for e in events]
    deploys = [e for e in event_list if e["kind"] in ("deploy", "pipeline", "config")]
    nearest_deploy = min(deploys, key=lambda e: abs(e["hours_from_onset"])) if deploys else None

    # performance before vs after onset (the AUC-vs-accuracy signature)
    pre = compute_window(db, model_id, end=onset)
    post = compute_window(db, model_id, start=onset)
    auc_delta = (pre["auc"] - post["auc"]) if pre and post and pre["auc"] and post["auc"] else None
    acc_delta = (pre["accuracy"] - post["accuracy"]) if pre and post else None

    top = [r.feature for r in flagged[:TOP_FEATURES]]
    shifts = _feature_shifts(model_id, top, onset, window_end or _now(), db)

    signals = {
        "max_psi": round(max((r.value for r in flagged), default=0.0), 3),
        "n_features_drifted": len(flagged),
        "prediction_drift_psi": round(pred_drift, 3) if pred_drift is not None else None,
        "auc_delta": round(auc_delta, 4) if auc_delta is not None else None,
        "accuracy_delta": round(acc_delta, 4) if acc_delta is not None else None,
        "nearest_deploy_hours": nearest_deploy["hours_from_onset"] if nearest_deploy else None,
        "feature_shifts": shifts,
    }

    return {
        "incident": {
            "id": incident.id, "model_id": model_id, "opened_ts": opened,
            "severity": incident.severity, "trigger_summary": incident.trigger_summary,
        },
        "drift": {
            "onset": onset, "window_end": window_end,
            "flagged_features": [
                {"report_id": r.id, "feature": r.feature, "metric": r.metric,
                 "psi": r.value, "threshold": r.threshold} for r in flagged
            ],
            "prediction_drift_psi": signals["prediction_drift_psi"],
        },
        "events": {"nearest_deploy": nearest_deploy, "within_48h": event_list},
        "performance": {"pre": pre, "post": post,
                        "auc_delta": signals["auc_delta"],
                        "accuracy_delta": signals["accuracy_delta"]},
        "signals": signals,
        "signature": classify_signature(signals),
    }
