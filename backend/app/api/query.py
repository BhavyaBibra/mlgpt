"""Read APIs: model health, drift reports, incidents, events — what the
dashboard and agent will consume."""
import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.agent import generate_explanation
from app.core.config import get_db
from app.core.context import assemble_incident_context
from app.core.drift import save_reference
from app.core.performance import compute_window, performance_timeseries
from app.models.db import DriftReport, Event, Explanation, Incident, Prediction

router = APIRouter(tags=["query"])


@router.post("/models/{model_id}/reference")
async def upload_reference(model_id: str, file: UploadFile):
    """Upload training/reference data (CSV) that drift is measured against."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")
    save_reference(model_id, df)
    return {"model_id": model_id, "rows": len(df), "columns": list(df.columns)}


@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    rows = db.execute(
        select(
            Prediction.model_id,
            func.count(Prediction.id),
            func.max(Prediction.ts),
        ).group_by(Prediction.model_id)
    ).all()
    out = []
    for model_id, count, last_ts in rows:
        open_incident = db.execute(
            select(Incident).where(Incident.model_id == model_id, Incident.status == "open")
        ).scalar_one_or_none()
        status = "degraded" if open_incident and open_incident.severity == "critical" \
            else "drifting" if open_incident else "healthy"
        out.append({
            "model_id": model_id, "predictions": count,
            "last_prediction": last_ts, "status": status,
        })
    return out


@router.get("/models/{model_id}/drift")
def get_drift(model_id: str, limit: int = 200, db: Session = Depends(get_db)):
    rows = db.execute(
        select(DriftReport)
        .where(DriftReport.model_id == model_id)
        .order_by(desc(DriftReport.window_end)).limit(limit)
    ).scalars().all()
    return [
        {
            "feature": r.feature, "metric": r.metric, "value": r.value,
            "threshold": r.threshold, "flagged": r.flagged,
            "window_start": r.window_start, "window_end": r.window_end,
        }
        for r in rows
    ]


@router.get("/incidents")
def list_incidents(status: str | None = None, db: Session = Depends(get_db)):
    q = select(Incident).order_by(desc(Incident.opened_ts))
    if status:
        q = q.where(Incident.status == status)
    return [
        {
            "id": i.id, "model_id": i.model_id, "opened_ts": i.opened_ts,
            "severity": i.severity, "status": i.status,
            "trigger_summary": i.trigger_summary,
        }
        for i in db.execute(q).scalars().all()
    ]


@router.get("/incidents/{incident_id}/context")
def incident_context(incident_id: int, db: Session = Depends(get_db)):
    """The assembled evidence packet + rules-based signature for an incident."""
    ctx = assemble_incident_context(db, incident_id)
    if ctx is None:
        raise HTTPException(404, "incident not found")
    return ctx


def _serialize_explanation(e: Explanation) -> dict:
    return {
        "id": e.id, "incident_id": e.incident_id, "ts": e.ts,
        "summary": e.summary, "root_cause": e.root_cause, "evidence": e.evidence,
        "confidence": e.confidence, "suggested_action": e.suggested_action,
    }


@router.post("/incidents/{incident_id}/explain")
def explain_incident(incident_id: int, db: Session = Depends(get_db)):
    """Generate + store the cited root-cause explanation. Idempotent: an incident
    keeps its first explanation (deterministic, auditable history)."""
    existing = db.execute(
        select(Explanation).where(Explanation.incident_id == incident_id)
        .order_by(Explanation.id).limit(1)
    ).scalar_one_or_none()
    if existing:
        return _serialize_explanation(existing)

    ctx = assemble_incident_context(db, incident_id)
    if ctx is None:
        raise HTTPException(404, "incident not found")
    result = generate_explanation(ctx)
    row = Explanation(
        incident_id=incident_id, summary=result["summary"],
        root_cause=result["root_cause"], evidence=result["evidence"],
        confidence=result["confidence"], suggested_action=result["suggested_action"],
    )
    db.add(row)
    db.commit()
    return _serialize_explanation(row) | {"_source": result.get("_source")}


@router.get("/incidents/{incident_id}/explanation")
def get_explanation(incident_id: int, db: Session = Depends(get_db)):
    row = db.execute(
        select(Explanation).where(Explanation.incident_id == incident_id)
        .order_by(Explanation.id).limit(1)
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "no explanation yet; POST /explain first")
    return _serialize_explanation(row)


@router.get("/models/{model_id}/performance")
def get_performance(model_id: str, buckets: int = 12, db: Session = Depends(get_db)):
    """Accuracy/AUC timeline (labelled traffic only) + an overall summary."""
    return {
        "overall": compute_window(db, model_id),
        "timeline": performance_timeseries(db, model_id, buckets=buckets),
    }


@router.get("/models/{model_id}/events")
def list_events(model_id: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(Event).where(Event.model_id == model_id).order_by(desc(Event.ts)).limit(100)
    ).scalars().all()
    return [
        {"id": e.id, "ts": e.ts, "kind": e.kind, "description": e.description}
        for e in rows
    ]
