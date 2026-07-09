"""Ingestion endpoints: predictions, actuals, events, reference data."""
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import get_db
from app.models.db import Actual, Event, Prediction

router = APIRouter(prefix="/ingest", tags=["ingest"])


class PredictionIn(BaseModel):
    model_id: str
    features: dict
    prediction: float
    model_version: str = "unknown"
    ts: datetime | None = None


class PredictionBatchIn(BaseModel):
    predictions: list[PredictionIn] = Field(max_length=5000)


class ActualIn(BaseModel):
    prediction_id: int
    actual: float


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
            **({"ts": p.ts} if p.ts else {}),
        )
        for p in batch.predictions
    ]
    db.add_all(rows)
    db.commit()
    return {"ingested": len(rows), "ids": [r.id for r in rows]}


@router.post("/actuals")
def ingest_actuals(items: list[ActualIn], db: Session = Depends(get_db)):
    db.add_all([Actual(prediction_id=a.prediction_id, actual=a.actual) for a in items])
    db.commit()
    return {"ingested": len(items)}


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
