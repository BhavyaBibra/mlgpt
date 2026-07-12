"""Natural-language query — "Ask MLGPT" ("why did accuracy drop this week?").

The hard work is already done and stored: incidents carry diagnosed, cited
explanations. So a question becomes: parse the model + time scope, retrieve the
relevant incidents/explanations, and answer with citations to real rows.

Uses Groq when a key is set (validated citations, retry); otherwise a
deterministic answer composed from the most relevant stored explanation — so it
always answers and always cites, never hand-waves.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, LLM_MAX_RETRIES
from app.core.explanations import get_explanation
from app.core.performance import compute_window
from app.models.db import Incident, Prediction, QaLog

_SEVERITY_RANK = {"critical": 2, "warning": 1}


def _parse_since(question: str) -> tuple[timedelta, str]:
    q = question.lower()
    if "today" in q:
        return timedelta(days=1), "today"
    if "yesterday" in q:
        return timedelta(days=2), "the last 2 days"
    if "week" in q:
        return timedelta(days=7), "this week"
    if "month" in q:
        return timedelta(days=30), "this month"
    m = re.search(r"(\d+)\s*(day|hour)s?", q)
    if m:
        n = int(m.group(1))
        return (timedelta(days=n), f"the last {n} days") if m.group(2) == "day" \
            else (timedelta(hours=n), f"the last {n} hours")
    return timedelta(days=30), "recently"


def _resolve_model(db: Session, question: str, model_id: str | None) -> str | None:
    if model_id:
        return model_id
    known = [r[0] for r in db.execute(select(Prediction.model_id).distinct())]
    for k in known:  # name-drop in the question wins
        if k.lower() in question.lower():
            return k
    return known[0] if len(known) == 1 else None


def _retrieve(db: Session, model_id: str, since: timedelta) -> dict:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - since
    incidents = db.execute(
        select(Incident).where(Incident.model_id == model_id)
        .order_by(desc(Incident.opened_ts))
    ).scalars().all()
    in_scope = []
    for inc in incidents:
        ots = inc.opened_ts.replace(tzinfo=None) if inc.opened_ts.tzinfo else inc.opened_ts
        if ots >= cutoff:
            expl = get_explanation(db, inc.id)
            in_scope.append((inc, expl))
    return {"incidents": in_scope, "performance": compute_window(db, model_id)}


def _bundle(model_id: str, scope: str, retrieved: dict) -> tuple[dict, set]:
    incidents = []
    allowed: set = set()
    for inc, expl in retrieved["incidents"]:
        allowed.add(inc.id)
        item = {"incident_id": inc.id, "opened_ts": str(inc.opened_ts),
                "severity": inc.severity, "trigger_summary": inc.trigger_summary}
        if expl:
            item["root_cause"] = expl.root_cause
            item["confidence"] = expl.confidence
            item["suggested_action"] = expl.suggested_action
            item["evidence"] = expl.evidence
            for e in expl.evidence or []:
                if e.get("ref_id") is not None:
                    allowed.add(e["ref_id"])
        incidents.append(item)
    return ({"model_id": model_id, "scope": scope, "incidents": incidents,
             "performance": retrieved["performance"]}, allowed)


# ---------- deterministic answer ----------

def _deterministic_answer(question: str, scope: str, bundle: dict) -> dict:
    incidents = bundle["incidents"]
    if not incidents:
        perf = bundle["performance"]
        acc = f" Latest measured accuracy is {perf['accuracy']}." if perf else ""
        return {"answer": f"No incidents for {bundle['model_id']} {scope}. "
                          f"The model looks healthy.{acc}",
                "citations": []}
    # most severe, then most recent
    inc = sorted(incidents, key=lambda i: (_SEVERITY_RANK.get(i["severity"], 0),
                                           i["opened_ts"]), reverse=True)[0]
    citations = [{"type": "incident", "ref": inc["incident_id"]}]
    for e in inc.get("evidence", []):
        if e.get("ref_id") is not None:
            citations.append({"type": e.get("type", "evidence"), "ref": e["ref_id"]})
    if "root_cause" in inc:
        answer = (f"{inc['root_cause']} "
                  f"(incident #{inc['incident_id']}, {inc['severity']}). "
                  f"Suggested action: {inc['suggested_action']} "
                  f"Confidence: {inc['confidence']}.")
    else:
        answer = (f"An incident opened {scope}: {inc['trigger_summary']} "
                  f"(incident #{inc['incident_id']}). No explanation stored yet.")
    return {"answer": answer, "citations": citations}


# ---------- LLM answer ----------

def _valid(ans: dict, allowed: set) -> bool:
    if not ans.get("answer"):
        return False
    cites = ans.get("citations", [])
    real = [c for c in cites if c.get("ref") in allowed]
    return len(real) >= 1


def _llm_answer(question: str, scope: str, bundle: dict, allowed: set) -> dict | None:
    system = (
        "You are MLGPT's analyst answering questions about ML model health from "
        "already-diagnosed incidents. Answer ONLY from the provided context. Return "
        "JSON {answer, citations} where citations is a list of {type, ref} and every "
        "ref MUST be an allowed id (incident id or evidence ref_id). Cite at least "
        "one. Be concise and specific; do not invent causes or ids."
    )
    user = (f"question: {question}\nscope: {scope}\nallowed_ids: {sorted(allowed)}\n\n"
            f"context:\n{json.dumps(bundle, default=str, indent=2)}")
    import httpx
    correction = None
    for _ in range(LLM_MAX_RETRIES + 1):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if correction:
            msgs.append({"role": "user", "content": f"Rejected: {correction}. Fix it."})
        try:
            r = httpx.post(f"{GROQ_BASE_URL}/chat/completions",
                           headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                           json={"model": GROQ_MODEL, "messages": msgs, "temperature": 0.2,
                                 "max_tokens": 700, "response_format": {"type": "json_object"}},
                           timeout=45)
            r.raise_for_status()
            out = json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            correction = f"call failed: {exc}"
            continue
        if _valid(out, allowed):
            return out
        correction = "citations must reference allowed ids"
    return None


def ask(db: Session, question: str, model_id: str | None = None,
        session_id: str | None = None) -> dict:
    model_id = _resolve_model(db, question, model_id)
    if model_id is None:
        return {"answer": "Which model do you mean? Please pass model_id.",
                "citations": [], "model_id": None, "scope": None}

    since, scope = _parse_since(question)
    bundle, allowed = _bundle(model_id, scope, _retrieve(db, model_id, since))

    result = None
    if GROQ_API_KEY:
        result = _llm_answer(question, scope, bundle, allowed)
    if result is None:
        result = _deterministic_answer(question, scope, bundle)

    db.add(QaLog(session_id=session_id, model_id=model_id, question=question,
                 answer=result["answer"], citations=result.get("citations", [])))
    db.commit()
    return {**result, "model_id": model_id, "scope": scope}
