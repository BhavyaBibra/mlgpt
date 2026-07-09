# MLGPT — 20-Day Build Plan

Assumes ~2–4 focused hours/day alongside your job/DSA prep. Each day ends with something demonstrable (which feeds the LinkedIn post for that day).

## Phase 1 — Foundation (Days 1–5)

**Day 1 — Repo + skeleton**
- Init repo, MIT license, project structure (backend/, frontend/, sdk/, docs/)
- Docker compose skeleton: Postgres + FastAPI hello-world
- Write the README problem statement (yes, day 1 — README-driven development keeps scope honest)

**Day 2 — Schema + SDK**
- Postgres schema (all 6 tables from ARCHITECTURE.md)
- `mlgpt` SDK: init/log_prediction/log_actual/log_event with async batching
- FastAPI ingest endpoints

**Day 3 — Traffic simulator**
- Pick dataset (LendingClub loan default or IEEE-CIS fraud)
- `mlgpt demo`: trains a quick baseline model, replays rows as timestamped live traffic
- Drift injection: at a chosen point, shift a feature distribution + degrade labels (this scripted "incident" is what every demo and the GIF will use)

**Day 4 — Drift detection**
- APScheduler job running Evidently suites on rolling windows vs. reference
- Write drift_reports; threshold config; open incidents on breach

**Day 5 — Milestone 1 ✅**
- End-to-end check: `docker compose up` → `mlgpt demo` → drift injected → incident row appears
- Ground-truth join + accuracy-over-time computation working

## Phase 2 — The Agent (Days 6–10)

**Day 6 — Context assembler**
- Given an incident: gather drift reports, ±48h events, pre/post statistical summaries, performance metrics into one context object

**Day 7 — Root-cause generation**
- LLM call with structured output schema (summary/root_cause/evidence/confidence/action)
- Citation validation in code: reject/retry outputs that don't reference real drift_report/event rows

**Day 8 — Explanation storage + alerting**
- Store explanations; Slack webhook with summary + deep link; dedup/cooldown logic

**Day 9 — NL query backend**
- Question → parse model/time scope → retrieve incidents/explanations/reports/events → cited answer
- Session history

**Day 10 — Milestone 2 ✅**
- Full loop test: injected drift → alert in Slack with explanation → ask "why did accuracy drop?" in a REPL → cited answer
- Tune the agent prompt on 5–10 synthetic incident scenarios (vary: feature drift only, deploy-caused, data-quality null spike, gradual concept drift)

## Phase 3 — Dashboard (Days 11–15)

**Day 11 — Frontend scaffold + overview**
- Vite + React + TS + Tailwind, dark theme tokens, layout shell, health cards with live WebSocket pulse

**Day 12 — Model detail: charts**
- Accuracy timeline with deploy markers + incident shading (Recharts)
- Feature drift heatmap (CSS grid)

**Day 13 — Incident view**
- Timeline component; explanation card; **clickable evidence chips → chart highlight** (budget the full day, this is the wow feature)

**Day 14 — Ask MLGPT panel**
- Chat UI, streaming answers, citation chips, suggested questions

**Day 15 — Milestone 3 ✅**
- Distribution overlay + time scrubber; events/settings screens; polish pass (spacing, empty states, loading states)

## Phase 4 — Launch prep (Days 16–20)

**Day 16 — Hardening**
- Clean-machine test of docker compose (borrow a friend's laptop); fix onboarding friction ruthlessly
- Error handling, sensible defaults, `.env.example`

**Day 17 — Docs**
- Quickstart, SDK reference, "How the agent works" deep-dive doc, 3–5 ADRs

**Day 18 — Milestone 4 ✅ README + GIF**
- Record the demo GIF: drift injected → alert → explanation → chat question → cited answer (rehearse; keep under 30s)
- Final README: story, GIF, quickstart, architecture diagram, honest comparison table, roadmap (auto-retraining, more detectors → invites contributors)

**Day 19 — Soft launch**
- Share with 5–10 people for feedback; fix top issues
- Prepare launch posts (HN Show HN draft, Reddit drafts, LinkedIn finale)

**Day 20 — Milestone 5 ✅ Public launch**
- Post: Show HN, r/mlops, r/MachineLearning, MLOps Slack, LinkedIn finale, X
- Respond to every comment fast for the first 48h (this materially affects traction)

## Risk buffers
- If the agent quality is poor by Day 8: narrow to deploy-correlated incidents only (still impressive, more reliable)
- If frontend runs over: cut the time scrubber and command palette, never cut the evidence chips
- If Spark mode fights you: ship simple mode, keep Spark as a documented optional flag (the code existing in the repo still signals the skill)
