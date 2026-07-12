"""Explanation service — assemble context, generate the cited explanation, and
store it exactly once.

Shared by the API (`POST /incidents/{id}/explain`) and the alerting path (auto-
explain when an incident opens) so both go through the same store-once logic:
an incident keeps its first explanation forever (deterministic, auditable).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.agent import generate_explanation
from app.core.context import assemble_incident_context
from app.models.db import Explanation


def get_explanation(db: Session, incident_id: int) -> Explanation | None:
    return db.execute(
        select(Explanation).where(Explanation.incident_id == incident_id)
        .order_by(Explanation.id).limit(1)
    ).scalar_one_or_none()


def ensure_explanation(db: Session, incident_id: int) -> tuple[Explanation | None, bool, str | None]:
    """Return (explanation, created, source). Idempotent: reuses the stored one.
    Returns (None, False, None) if the incident doesn't exist."""
    existing = get_explanation(db, incident_id)
    if existing:
        return existing, False, None

    ctx = assemble_incident_context(db, incident_id)
    if ctx is None:
        return None, False, None
    result = generate_explanation(ctx)
    row = Explanation(
        incident_id=incident_id, summary=result["summary"],
        root_cause=result["root_cause"], evidence=result["evidence"],
        confidence=result["confidence"], suggested_action=result["suggested_action"],
    )
    db.add(row)
    db.commit()
    return row, True, result.get("_source")
