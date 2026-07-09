# MLGPT — Project Context for Claude Code

## What this is
MLGPT is an open-source ML observability agent: it detects model drift in
production and explains the ROOT CAUSE in plain English, with enforced
citations to real evidence. Tagline: "Datadog told you it broke. MLGPT tells
you why."

Full plans live in `docs/`: PLAN.md (vision/scope), ARCHITECTURE.md,
UI.md (dashboard design), BUILD_PLAN.md (20-day schedule),
LINKEDIN_CALENDAR.md (build-in-public posts). Read PLAN.md and
ARCHITECTURE.md before making changes.

## Positioning rule (never violate)
We build ON Evidently for drift statistics — never reimplement drift math.
MLGPT's value is the intelligence layer: correlation, explanation, NL query.

## Status: Phase 1 COMPLETE ✅ (tested end-to-end)
- 6-table Postgres schema (backend/app/models/db.py)
- FastAPI ingest + query APIs
- SDK (sdk/mlgpt): batched log_prediction / log_actual / log_event / upload_reference
- PSI drift engine + APScheduler loop + incident auto-open with dedup (core/drift.py)
- Demo simulator (scripts/demo.py): replays traffic, logs deploy event at 40%,
  injects transaction_amount drift at 45% — verified: flags the right feature
  (PSI 0.29), opens incident, other features stay green
- docker-compose.yml: postgres + backend

## Known gotchas
- Use 127.0.0.1 not localhost in clients (httpx resolves localhost to ::1,
  uvicorn binds IPv4)
- python-multipart is required for the reference upload endpoint (in requirements)
- Reference data stored as parquet in REFERENCE_DIR (default /tmp/mlgpt_reference)
- Drift check interval/threshold/window are env vars (see core/config.py)

## Next: Phase 2 (Days 6–10 in docs/BUILD_PLAN.md)
1. Context assembler: for an incident, gather drift reports + events within
   ±48h + pre/post statistical summaries + performance metrics
2. Root-cause agent: LLM call with STRUCTURED output schema
   (summary/root_cause/evidence/confidence/suggested_action).
   HARD RULE: every explanation must cite ≥1 real drift_report row and,
   where applicable, an event row — validate in code, reject/retry otherwise.
   This citation enforcement is the project's core differentiator.
3. Store explanations (never regenerate — deterministic, auditable history)
4. Slack webhook alerts with explanation attached, cooldown/dedup
5. NL query endpoint: question → parse model/time scope → retrieve
   incidents/explanations/reports/events → cited answer
Then Phase 3: React+TS dark-mode dashboard per docs/UI.md (priority feature:
clickable evidence chips that highlight the exact chart region cited).

## Conventions
- Python 3.11+, type hints, SQLAlchemy 2.0 style
- Keep the demo (`python scripts/demo.py --fast`) working after every change —
  it's the acceptance test and the LinkedIn demo material
- Owner is posting daily build-in-public updates; each work session should
  end with something demonstrable
