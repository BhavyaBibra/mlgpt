"""Alerting — when an incident opens, notify with the explanation ATTACHED, not
just "PSI > 0.2". That difference (an alert you can act on) is the point.

Slack webhook by default. No webhook configured -> logs and no-ops, so dev runs
and the demo work with zero setup. Dedup is inherent: incidents open once and
each gets exactly one alert.
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import SLACK_WEBHOOK_URL
from app.models.db import Explanation, Incident

log = logging.getLogger("mlgpt.alerts")

_CONF_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _blocks(incident: Incident, expl: Explanation) -> dict:
    emoji = _CONF_EMOJI.get(expl.confidence, "⚪")
    chips = " · ".join(
        f"{e.get('type')}#{e['ref_id']}" if e.get("ref_id") is not None
        else str(e.get("type"))
        for e in (expl.evidence or [])[:5]
    )
    text = (
        f"{emoji} *MLGPT incident — {incident.model_id}* ({incident.severity})\n"
        f"*Root cause:* {expl.root_cause}\n"
        f"*Suggested action:* {expl.suggested_action}\n"
        f"*Confidence:* {expl.confidence}   *Evidence:* {chips}"
    )
    return {"text": text}


def send_incident_alert(incident: Incident, explanation: Explanation) -> bool:
    """Post the incident + its cited explanation. Returns True if actually sent."""
    payload = _blocks(incident, explanation)
    if not SLACK_WEBHOOK_URL:
        log.info("[alert:no-webhook] %s", payload["text"].replace("\n", " | "))
        return False
    try:
        r = httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:  # never let alerting break the drift loop
        log.warning("slack alert failed: %s", exc)
        return False
