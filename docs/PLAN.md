# MLGPT — The ML Model Health Agent

> **One-liner:** Your ML model is degrading in production right now. MLGPT tells you *why* — in plain English.

**Tagline options** (pick one for the repo header):
- "Datadog told you it broke. MLGPT tells you why."
- "Ask your models what's wrong with them."
- "ChatGPT for your production ML models."

---

## 1. The Problem (this goes at the top of the README)

A fraud model launches at 97% accuracy. Three months later it's at 82%. Nobody noticed. Fraud losses climbed.

The model didn't break — the world changed. Customers changed habits, a feature went null, a pipeline silently changed format.

Existing tools (Arize, Evidently, Fiddler) show you **dashboards of what drifted**. MLGPT is the layer on top: an **agent that investigates the drift, correlates it with your deploy history, and explains the root cause in plain English** — and lets you ask questions like *"why did accuracy drop this week?"*

## 2. Positioning (critical for both GitHub and interviews)

We do NOT compete with Evidently/Arize on drift math. We **use Evidently** for statistics and build the intelligence layer they're still maturing:

| Layer | Who owns it |
|---|---|
| Drift statistics (PSI, KL, JS) | Evidently (dependency) |
| Streaming ingestion | **MLGPT** (Spark Structured Streaming) |
| Event timeline store | **MLGPT** (Postgres) |
| Root-cause agent | **MLGPT** ← the star of the show |
| Natural-language Q&A | **MLGPT** |
| Live dashboard | **MLGPT** |

Interview framing: *"I knew Arize/Evidently exist — I used Evidently for detection and built the explanation layer on top. The hard problem isn't detecting drift, it's answering 'so what, and why?'"*

## 3. Core Features (v1.0 scope — everything here ships)

### 3.1 Ingestion
- Prediction logging SDK: `mlgpt.log_prediction(features, prediction, model_version)` — one line to instrument any model
- Ground-truth join: `mlgpt.log_actual(prediction_id, actual)` — links outcomes back to predictions
- Deploy-event hook: `mlgpt.log_event("deployed v2.3")` or auto-capture via git hook
- Traffic simulator included (`mlgpt demo`) so anyone can see it working in 60 seconds without a real model — **this is essential for GitHub adoption**

### 3.2 Detection (via Evidently, scheduled)
- Data drift per feature (PSI, JS distance)
- Prediction drift (output distribution shift)
- Performance decay (accuracy/F1/precision vs. ground truth as it arrives)
- Data-quality checks (null spikes, new categories, schema changes)

### 3.3 The Agent (the differentiator)
When a check fires, the agent:
1. Pulls the drift report (which features, how much, since when)
2. Pulls the event timeline around the drift window (deploys, pipeline changes)
3. Pulls sample rows from before/after the shift
4. Produces a **structured root-cause report**: what happened → most likely cause → evidence → suggested action
5. Every claim must cite a feature, timestamp, or event — no vague hand-waving (enforced via structured output schema)

### 3.4 Ask-Your-Model (NL query)
- Chat box in the dashboard: "why did fraud accuracy drop this week?"
- Agent retrieves relevant timeline slice + drift reports + past incident explanations, answers with citations to specific events
- Past explanations are stored — answers are consistent, not regenerated randomly

### 3.5 Alerts
- Slack webhook + email
- Alert = explanation attached, not just "PSI > 0.2"
- Severity levels: 🟢 healthy / 🟡 drifting / 🔴 degraded

### 3.6 Dashboard (see UI.md for full design)
- Model health overview (status cards per model)
- Per-feature drift heatmap over time
- Accuracy timeline with deploy markers annotated on the chart
- Incident timeline (deploy → drift detected → explanation → resolved)
- Chat panel for NL questions

## 4. Explicitly OUT of scope for v1 (say no, ship faster)
- Kubernetes / Helm charts (add later if traction)
- Auto-retraining pipelines (v2 roadmap item — great "coming soon" for README)
- Multi-tenancy / auth beyond a simple API key
- Canary deployments
- Reimplementing drift statistics

## 5. Tech Stack (final)

| Layer | Choice | Why |
|---|---|---|
| Ingestion | Spark Structured Streaming (with a lightweight Python-only fallback mode) | Your real Databricks strength; fallback = easy local demo |
| Storage | PostgreSQL | Predictions, events, drift reports, explanations |
| Drift | Evidently AI | Battle-tested, open source, respected |
| Agent | Claude/OpenAI API via LangChain (or raw API) with structured outputs | Root cause + NL query |
| Backend API | FastAPI | Fast, typed, familiar to reviewers |
| Frontend | React + TypeScript + Recharts + Tailwind | See UI.md |
| Alerts | Slack webhook, SMTP | Simple, demo-friendly |
| Packaging | Docker Compose (one command: `docker compose up`) | Zero-friction trial is non-negotiable for stars |
| Demo | `mlgpt demo` replays LendingClub/fraud dataset as live traffic | The 60-second wow |

## 6. GitHub Star Strategy (what actually earns stars)

Stars come from: (1) a killer README, (2) a 60-second try-it experience, (3) a great demo GIF, (4) distribution. Plan for each:

1. **README** — problem story first (the 97%→82% fraud model), animated GIF of the agent explaining an incident, `docker compose up && mlgpt demo` quickstart, architecture diagram, honest comparison table vs Evidently/Arize ("we build ON Evidently, not against it" — this earns respect, not competition)
2. **The demo GIF** — screen recording: drift injected → alert fires → agent explanation appears → user asks "why?" in chat → cited answer. This GIF is the single highest-ROI asset in the whole project.
3. **Zero-friction trial** — `docker compose up` must work first try on a clean machine. Test on a friend's laptop.
4. **Distribution** — launch posts on: r/MachineLearning, r/mlops, Hacker News (Show HN), LinkedIn (your 14-day series builds the audience *before* launch), MLOps Community Slack, X/Twitter ML circles. Launch AFTER the demo GIF is perfect, not before.
5. **Docs** — a `docs/` folder with: quickstart, SDK reference, "how the agent works" (people love reading agent internals), architecture decision records.

Honest expectation: a polished, useful tool with good distribution typically lands 100–2,000 stars in the first months. Viral outliers exist but can't be engineered on demand. Either outcome is a strong resume line.

## 7. Resume Lines (draft now, refine after shipping)

- "Built MLGPT, an open-source ML observability agent that performs automated root-cause analysis on model drift, correlating statistical drift signals with deployment history to generate cited, plain-English incident explanations."
- "Designed streaming ingestion (Spark Structured Streaming → Postgres) processing simulated production traffic of N predictions/day with scheduled Evidently-based drift detection."
- "Implemented a natural-language query interface over model health history; [X] GitHub stars, [Y] Docker pulls." *(fill in real numbers only)*

## 8. Milestones

- **M1 (Day 5):** ingestion + storage + drift detection working end to end, CLI demo replaying data
- **M2 (Day 10):** agent generating cited root-cause reports; NL query answering from timeline
- **M3 (Day 15):** dashboard live with all views; alerts firing with explanations
- **M4 (Day 18):** README + GIF + docs polished; docker compose tested clean
- **M5 (Day 20):** public launch (HN, Reddit, LinkedIn finale post)

Full day-by-day build order: see BUILD_PLAN.md
UI design: see UI.md
Architecture detail: see ARCHITECTURE.md
LinkedIn calendar: see LINKEDIN_CALENDAR.md
