# MLGPT Architecture

## High-level flow

```
                          ┌──────────────────────────┐
                          │        Your Model         │
                          │  (or `mlgpt demo` replay) │
                          └──────┬──────────┬────────┘
                                 │          │
                    log_prediction()   log_event("deployed v2.3")
                                 │          │
                          ┌──────▼──────────▼────────┐
                          │      Ingestion Layer      │
                          │  Spark Structured Streaming│
                          │  (fallback: FastAPI direct)│
                          └────────────┬──────────────┘
                                       │
                          ┌────────────▼──────────────┐
                          │        PostgreSQL          │
                          │ predictions │ actuals      │
                          │ events      │ drift_reports│
                          │ incidents   │ explanations │
                          └───┬─────────────────┬─────┘
                              │                 │
               ┌──────────────▼───┐   ┌─────────▼──────────────┐
               │  Drift Scheduler  │   │     FastAPI Backend     │
               │ (Evidently, cron) │   │  REST + WebSocket       │
               └──────────┬───────┘   └─────────┬──────────────┘
                          │ drift flagged        │
               ┌──────────▼───────┐              │
               │  Root-Cause Agent │              │
               │  (LLM, structured │              │
               │   output, cited)  │              │
               └──────┬───────────┘              │
                      │                          │
            ┌─────────▼─────────┐      ┌─────────▼─────────┐
            │   Alert Manager    │      │  React Dashboard  │
            │  (Slack / email)   │      │  (charts + chat)  │
            └───────────────────┘      └───────────────────┘
```

## Components

### 1. SDK (`mlgpt` Python package)
```python
import mlgpt

mlgpt.init(api_url="http://localhost:8000", model_name="fraud-v2")

# In your prediction path (one line):
mlgpt.log_prediction(features=row, prediction=pred, model_version="2.3")

# When ground truth arrives:
mlgpt.log_actual(prediction_id=pid, actual=1)

# On deploy (or via git hook):
mlgpt.log_event("Deployed model v2.3", kind="deploy")
```
Batched async HTTP under the hood — near-zero latency added to the caller.

### 2. Ingestion
- **Primary path:** SDK → FastAPI ingest endpoint → Kafka-less design: writes to a Postgres staging table; Spark Structured Streaming job reads staging → validates → writes to `predictions`
- **Simple mode** (default in docker compose): FastAPI writes directly to Postgres. Spark mode is a flag — keeps the demo light but shows real streaming chops in the repo
- Traffic simulator: `mlgpt demo --dataset lending` replays a public dataset with injected drift at a known timestamp (so the demo always has a story to tell)

### 3. Storage schema (Postgres)
```sql
predictions(id, model_id, ts, features JSONB, prediction, model_version)
actuals(prediction_id, actual, ts)
events(id, model_id, ts, kind, description)            -- deploys, config changes
drift_reports(id, model_id, window_start, window_end,
              feature, metric, value, threshold, flagged)
incidents(id, model_id, opened_ts, severity, status)
explanations(id, incident_id, ts, summary, evidence JSONB, suggested_action)
```

### 4. Drift Scheduler
- APScheduler job (every N minutes): pull last window of predictions, run Evidently suite vs. reference data
- Writes `drift_reports`; when thresholds breached → opens an `incident` → triggers the agent

### 5. Root-Cause Agent (the core IP)
Input context assembled per incident:
- Drift report (features, metrics, magnitude, window)
- Events within ±48h of drift onset
- Sample rows pre/post shift (statistical summary, not raw PII)
- Model performance metrics around the window

Output: **structured JSON, validated against a schema**:
```json
{
  "summary": "Accuracy dropped 6.2% starting 2026-07-08 14:00",
  "root_cause": "Distribution shift in transaction_amount coinciding with v2.3 deploy",
  "evidence": [
    {"type": "drift", "feature": "transaction_amount", "psi": 0.31, "window": "..."},
    {"type": "event", "event_id": 42, "description": "Deployed v2.3", "ts": "..."}
  ],
  "confidence": "high",
  "suggested_action": "Compare v2.3 preprocessing of transaction_amount against v2.2"
}
```
Rule: every explanation must reference at least one drift_report row and, where applicable, an event row — enforced in code, and rendered as clickable citations in the UI. This is what makes it credible instead of LLM hand-waving.

### 6. NL Query ("Ask your model")
- Retrieval over: incidents, explanations, drift_reports, events (filtered by model + time range parsed from the question)
- Answer generated with the same citation rule
- Conversation history stored per session

### 7. Alert Manager
- On incident open: Slack webhook + optional email with the agent's summary + dashboard deep link
- Cooldown/dedup so one drift episode = one alert, not fifty

### 8. Frontend (see UI.md)
- React + TS, talks to FastAPI REST + WebSocket (live updates)

## Key design decisions (write these up as ADRs in the repo — reviewers love them)
1. **Build on Evidently, don't reimplement statistics** — credibility + speed
2. **Citations enforced by schema, not prompt hope** — the difference between a toy and a tool
3. **Simple mode default, Spark mode optional** — frictionless demo AND real streaming credibility
4. **Explanations are stored, not regenerated** — deterministic history, auditable
5. **No Kubernetes in v1** — `docker compose up` is the whole install story
