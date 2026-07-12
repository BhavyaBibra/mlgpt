"""Root-cause agent — turns the cited evidence packet into a plain-English,
structured explanation.

Division of labour that keeps this trustworthy:
  - `context.classify_signature` already decided WHAT went wrong (deploy /
    data-quality / world-change) deterministically from the evidence.
  - the LLM only PHRASES that, and must cite real drift_report / event ids.

Guarantees:
  - every explanation cites >=1 real row (validated in code; reject + retry).
  - if the LLM is unavailable, keeps failing, or no key is set, a deterministic
    explanation is built from the signature — so the system NEVER hand-waves and
    NEVER blocks. The model can make the wording nicer; it cannot invent a cause.
"""
from __future__ import annotations

import json

import httpx

from app.core.config import (
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, LLM_MAX_RETRIES,
)

_LABEL_TEXT = {
    "data_quality": "a data-quality break (a feature went missing or collapsed to one value)",
    "deploy_or_pipeline": "a shipped change that corrupted a model input",
    "world_change": "a genuine shift in the population/world (not a pipeline bug)",
    "ambiguous": "an unclear cause that needs a human look",
}
_LABEL_ACTION = {
    "data_quality": "Fix the upstream pipeline for {feat}; backfill/repair the affected "
                    "rows{deploy}. Do not retrain until inputs are clean.",
    "deploy_or_pipeline": "Compare {feat} preprocessing against the previous model "
                          "version{deploy}; roll back if confirmed.",
    "world_change": "Retrain on recent labelled data and re-validate feature relevance; "
                    "there is no pipeline bug to roll back.",
    "ambiguous": "Investigate manually — the signals are mixed and don't attribute cleanly.",
}


# ---------- citations ----------

def _allowed_ref_ids(context: dict) -> set[int]:
    ids = {f["report_id"] for f in context["drift"]["flagged_features"]}
    ids |= {n["report_id"] for n in context["drift"].get("null_spikes", [])}
    ids |= {e["event_id"] for e in context["events"]["within_48h"]}
    if context["performance"].get("report_id") is not None:
        ids.add(context["performance"]["report_id"])
    return ids


def _validate(explanation: dict, allowed: set[int]) -> tuple[bool, str]:
    ev = explanation.get("evidence")
    if not isinstance(ev, list) or not ev:
        return False, "no evidence array"
    cited_real = 0
    for item in ev:
        ref = item.get("ref_id")
        if item.get("type") == "performance" and ref is None:
            continue
        if ref is None:
            return False, "evidence item missing ref_id"
        if ref not in allowed:
            return False, f"evidence cites unknown ref_id {ref}"
        cited_real += 1
    if cited_real < 1:
        return False, "no citation to a real drift/event row"
    for k in ("summary", "root_cause", "confidence", "suggested_action"):
        if not explanation.get(k):
            return False, f"missing {k}"
    return True, "ok"


# ---------- deterministic fallback (always valid, always cited) ----------

def _deterministic(context: dict) -> dict:
    sig = context["signature"]
    label = sig["label"]
    flagged = context["drift"]["flagged_features"]
    nulls = context["drift"].get("null_spikes", [])
    deploy = context["events"]["nearest_deploy"]
    perf = context["performance"]

    top_feat = (flagged[0]["feature"] if flagged
                else nulls[0]["feature"] if nulls else "the affected feature")

    evidence = []
    for f in flagged[:4]:
        evidence.append({"type": "drift", "ref_id": f["report_id"],
                         "detail": f"{f['feature']} {f['metric']} {f['value']}"})
    for n in nulls[:2]:
        evidence.append({"type": "data_quality", "ref_id": n["report_id"],
                         "detail": f"{n['feature']} null rate {n['null_rate']}"})
    if deploy:
        evidence.append({"type": "event", "ref_id": deploy["event_id"],
                         "detail": f"{deploy['description']} ({deploy['hours_from_onset']:+.1f}h from onset)"})
    if perf.get("pre") and perf.get("post"):
        evidence.append({"type": "performance", "ref_id": perf.get("report_id"),
                         "detail": f"accuracy {perf['pre']['accuracy']} -> {perf['post']['accuracy']}, "
                                   f"AUC {perf['pre']['auc']} -> {perf['post']['auc']}"})

    deploy_clause = f" (introduced by '{deploy['description']}')" if deploy and label != "world_change" else ""
    action = _LABEL_ACTION[label].format(feat=top_feat, deploy=deploy_clause)

    acc_bit = ""
    if perf.get("pre") and perf.get("post"):
        acc_bit = f" Accuracy moved {perf['pre']['accuracy']} -> {perf['post']['accuracy']}."

    return {
        "summary": f"{context['incident']['trigger_summary']}.{acc_bit}",
        "root_cause": f"Most likely {_LABEL_TEXT[label]}. {sig['rationale']}",
        "evidence": evidence,
        "confidence": sig["confidence"],
        "suggested_action": action,
        "_source": "deterministic",
    }


# ---------- LLM path ----------

def _compact_context(context: dict) -> dict:
    """Just the fields the model needs, with the real ref_ids it must cite."""
    return {
        "incident": context["incident"],
        "signature_prior": context["signature"],
        "flagged_features": context["drift"]["flagged_features"],
        "null_spikes": context["drift"].get("null_spikes", []),
        "nearest_deploy": context["events"]["nearest_deploy"],
        "performance": context["performance"],
        "feature_shifts": context["signals"]["feature_shifts"][:4],
    }


def _messages(context: dict, correction: str | None = None) -> list[dict]:
    allowed = sorted(_allowed_ref_ids(context))
    system = (
        "You are MLGPT's root-cause analyst for ML model incidents. A rules engine "
        "has already classified the cause (signature_prior) from the evidence — trust "
        "it unless the evidence plainly contradicts it. Your job is to explain it "
        "clearly, not to invent a new cause.\n"
        "Return ONLY a JSON object: {summary, root_cause, evidence, confidence, "
        "suggested_action}. `evidence` is a list of {type, ref_id, detail}; every "
        "ref_id MUST be one of the allowed_ref_ids from the context (except a single "
        "optional performance item may use ref_id null). Cite at least one real row. "
        "Do not invent ids or causes. confidence is one of high|medium|low and should "
        "match signature_prior. Be concise and specific (name the feature, the deploy, "
        "the accuracy move)."
    )
    user = (f"allowed_ref_ids: {allowed}\n\ncontext:\n"
            + json.dumps(_compact_context(context), default=str, indent=2))
    if correction:
        user += f"\n\nYour previous answer was rejected: {correction}. Fix it."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_groq(messages: list[dict]) -> dict:
    resp = httpx.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.2,
              "max_tokens": 900, "response_format": {"type": "json_object"}},
        timeout=45,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def generate_explanation(context: dict) -> dict:
    """Cited explanation for an assembled incident context. Uses Groq when a key
    is set and its output validates; otherwise the deterministic fallback."""
    if not GROQ_API_KEY:
        return _deterministic(context)

    allowed = _allowed_ref_ids(context)
    correction = None
    for _ in range(LLM_MAX_RETRIES + 1):
        try:
            out = _call_groq(_messages(context, correction))
        except Exception as exc:  # network / parse / API error -> fall back
            correction = f"call failed: {exc}"
            continue
        ok, why = _validate(out, allowed)
        if ok:
            out["_source"] = "llm"
            # confidence stays anchored to the deterministic signature
            out.setdefault("confidence", context["signature"]["confidence"])
            return out
        correction = why
    return _deterministic(context)
