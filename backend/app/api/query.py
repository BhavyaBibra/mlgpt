"""Read APIs: model health, drift reports, incidents, events — what the
dashboard and agent will consume."""
import io

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.core.config import get_db
from app.core.drift import save_reference
from app.models.db import DriftReport, Event, Incident, Prediction

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


@router.get("/models/{model_id}/events")
def list_events(model_id: str, db: Session = Depends(get_db)):
    rows = db.execute(
        select(Event).where(Event.model_id == model_id).order_by(desc(Event.ts)).limit(100)
    ).scalars().all()
    return [
        {"id": e.id, "ts": e.ts, "kind": e.kind, "description": e.description}
        for e in rows
    ]
