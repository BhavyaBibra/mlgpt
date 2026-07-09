# MLGPT

> **Datadog told you it broke. MLGPT tells you why.**

Your fraud model launched at 97% accuracy. Three months later it's at 82%. Nobody noticed. Losses climbed.

The model didn't break — **the world changed.** Customers changed habits, a feature went null, a pipeline silently changed format.

Drift dashboards (Evidently, Arize, Fiddler) show you **what** drifted. MLGPT is the layer on top: an **agent that investigates drift, correlates it with your deploy history, and explains the root cause in plain English** — with citations to real evidence, not hand-waving.

```
You:    why did accuracy drop this week?
MLGPT:  Accuracy fell 6.2% starting Tue 14:40, driven by a distribution
        shift in `transaction_amount` (PSI 0.31) that began 38 minutes
        after the v2.3 deploy [event #42]. Suggested action: compare
        v2.3 preprocessing of transaction_amount against v2.2.
```

## Quickstart (60 seconds)

```bash
git clone https://github.com/<you>/mlgpt && cd mlgpt
docker compose up -d
pip install -e sdk/
python scripts/demo.py        # replays real data as live traffic, injects drift
```

Open http://localhost:8000/docs for the API. Dashboard coming in Phase 3.

## How it works

```
Your model ──log_prediction()──▶ Ingestion ──▶ Postgres ◀── deploy events
                                                  │
                                    Evidently drift checks (scheduled)
                                                  │ incident opened
                                        Root-cause agent (LLM,
                                        citation-enforced output)
                                                  │
                                     Slack alert + dashboard + chat
```

We deliberately **build on Evidently** for drift statistics rather than reimplementing them — MLGPT's job is the intelligence layer on top: correlation, explanation, and natural-language querying.

## Status

🚧 Building in public — follow along on [LinkedIn](#). Phase 1 (ingestion + detection) in progress.

## Roadmap
- [x] Prediction/event ingestion SDK
- [x] Drift detection (Evidently) + incident creation
- [ ] Root-cause agent with enforced citations
- [ ] Ask-your-model natural language queries
- [ ] Dashboard (dark mode, evidence chips → chart highlights)
- [ ] Auto-retraining recommendations (v2)

## License
MIT
