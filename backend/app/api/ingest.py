"""Ingestion endpoints: predictions, actuals, events, reference data."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_db
from app.models.db import Actual, Event, Prediction

router = APIRouter(prefix="/ingest", tags=["ingest"])


class PredictionIn(BaseModel):
    model_id: str
    features: dict
    prediction: float
    model_version: str = "unknown"
    ext_id: str | None = None
    ts: datetime | None = None


class PredictionBatchIn(BaseModel):
    predictions: list[PredictionIn] = Field(max_length=5000)


class ActualIn(BaseModel):
    actual: float
    # supply either the row id we returned, or the (model_id, ext_id) the client
    # tagged the prediction with
    prediction_id: int | None = None
    model_id: str | None = None
    ext_id: str | None = None

    @model_validator(mode="after")
    def _need_a_key(self):
        if self.prediction_id is None and not (self.model_id and self.ext_id):
            raise ValueError("actual needs prediction_id or (model_id + ext_id)")
        return self


class EventIn(BaseModel):
    model_id: str
    description: str
    kind: str = "note"
    ts: datetime | None = None


@router.post("/predictions")
def ingest_predictions(batch: PredictionBatchIn, db: Session = Depends(get_db)):
    rows = [
        Prediction(
            model_id=p.model_id,
            features=p.features,
            prediction=p.prediction,
            model_version=p.model_version,
            ext_id=p.ext_id,
            **({"ts": p.ts} if p.ts else {}),
        )
        for p in batch.predictions
    ]
    db.add_all(rows)
    db.commit()
    return {"ingested": len(rows), "ids": [r.id for r in rows]}


@router.post("/actuals")
def ingest_actuals(items: list[ActualIn], db: Session = Depends(get_db)):
    # resolve any (model_id, ext_id) keys to prediction ids in one pass
    keys = {(a.model_id, a.ext_id) for a in items if a.prediction_id is None}
    resolved: dict[tuple[str, str], int] = {}
    if keys:
        model_ids = {m for m, _ in keys}
        ext_ids = {e for _, e in keys}
        for pid, mid, ext in db.execute(
            select(Prediction.id, Prediction.model_id, Prediction.ext_id).where(
                Prediction.model_id.in_(model_ids), Prediction.ext_id.in_(ext_ids)
            )
        ):
            resolved[(mid, ext)] = pid  # last write wins if ext_id reused

    rows, unmatched = [], 0
    for a in items:
        pid = a.prediction_id or resolved.get((a.model_id, a.ext_id))
        if pid is None:
            unmatched += 1
            continue
        rows.append(Actual(prediction_id=pid, actual=a.actual))
    db.add_all(rows)
    db.commit()
    return {"ingested": len(rows), "unmatched": unmatched}


@router.post("/events")
def ingest_event(event: EventIn, db: Session = Depends(get_db)):
    row = Event(
        model_id=event.model_id,
        description=event.description,
        kind=event.kind,
        **({"ts": event.ts} if event.ts else {}),
    )
    db.add(row)
    db.commit()
    return {"id": row.id}
