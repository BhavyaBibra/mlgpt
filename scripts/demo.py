"""MLGPT demo (Option B) — a REAL model on REAL financial data, decaying live.

Trains a HistGradientBoosting model on the UCI credit-default dataset, uploads
its reference distribution, then replays real held-out applications as
timestamped production traffic:

  ~0-40%   normal traffic (model v2.2), ground truth logged            -> healthy
  40%      "deploy v2.3" event logged
  45%+     v2.3's preprocessing bug flattens the repayment-status block -> the
           model's strongest signal is corrupted; drift + accuracy decay follow

Within one drift-check interval MLGPT flags the pay_status_* features and opens
an incident; the ground-truth join then shows the accuracy/AUC drop that proves
it mattered. Phase 2's agent explains why.

Usage:
    python scripts/demo.py            # ~90s, live pacing
    python scripts/demo.py --fast     # no pacing, seconds

Needs the backend up (docker compose up) or DATABASE_URL pointing at a db.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))                       # scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
import creditdata as C
import mlgpt

MODEL_ID = "credit-default"
N_LIVE = 2000
SPAN_MINUTES = 90          # replay is stamped across this much recent time
DEPLOY_FRAC = 0.40
DRIFT_FRAC = 0.45


def _build_stream(model, feats, live_pool):
    """Assemble the ordered stream of (features, score, actual, regime)."""
    rng = np.random.default_rng(2024)
    deploy_at = int(N_LIVE * DEPLOY_FRAC)
    drift_at = int(N_LIVE * DRIFT_FRAC)

    Xb, yb, _ = C.regime_batch(model, feats, live_pool, "baseline", drift_at, rng)
    Xd, yd, _ = C.regime_batch(model, feats, live_pool, "deploy_bug",
                               N_LIVE - drift_at, rng)
    X = pd.concat([Xb, Xd], ignore_index=True)
    y = np.concatenate([yb, yd])
    proba = model.predict_proba(X)[:, 1]

    now = datetime.now(timezone.utc)
    t0 = now - timedelta(minutes=SPAN_MINUTES)
    step = timedelta(minutes=SPAN_MINUTES) / N_LIVE
    return X, y, proba, [t0 + step * i for i in range(N_LIVE)], deploy_at, drift_at


def main(fast: bool) -> None:
    delay = 0.0 if fast else 0.03

    print("→ Training the real model on UCI credit-default data…")
    model, reference, m, feats, live_pool = C.train_model()
    print(f"  baseline: accuracy {m['accuracy']*100:.1f}%  AUC {m['auc']:.3f}")

    print("→ Connecting to MLGPT and uploading reference distribution…")
    mlgpt.init(api_url="http://127.0.0.1:8000", model_id=MODEL_ID, model_version="2.2")
    ref_path = "/tmp/mlgpt_demo_reference.csv"
    reference.to_csv(ref_path, index=False)
    print("  ", mlgpt.upload_reference(ref_path))

    X, y, proba, ts, deploy_at, drift_at = _build_stream(model, feats, live_pool)
    print(f"→ Streaming {N_LIVE} real predictions + ground truth "
          f"(deploy at #{deploy_at}, drift at #{drift_at})…")

    pending_actuals: list[dict] = []
    for i in range(N_LIVE):
        if i == deploy_at:
            mlgpt.log_event("Deployed model v2.3 (rewrote repayment-status "
                            "preprocessing)", kind="deploy", ts=ts[i])
            print(f"\n  🚀 [{i}] deploy v2.3 logged\n")

        row = {k: float(v) for k, v in X.iloc[i].items()}
        version = "2.3" if i >= deploy_at else "2.2"
        mlgpt.log_prediction(features=row, prediction=float(proba[i]),
                             model_version=version, ext_id=str(i), ts=ts[i])
        pending_actuals.append({"actual": float(y[i]), "ext_id": str(i)})

        if len(pending_actuals) >= 500:
            mlgpt.flush()
            mlgpt.log_actuals(pending_actuals)
            pending_actuals = []
        if i % 250 == 0:
            print(f"  [{i}] streaming{' (v2.3 DRIFTED)' if i >= drift_at else ''}…")
        if delay:
            time.sleep(delay)

    mlgpt.flush()
    if pending_actuals:
        mlgpt.log_actuals(pending_actuals)

    print("\n✓ Demo traffic complete (predictions + ground truth ingested).")
    print("  Within one drift-check interval MLGPT flags pay_status_* and opens")
    print("  an incident. Inspect:")
    print(f"    curl 127.0.0.1:8000/incidents")
    print(f"    curl 127.0.0.1:8000/models/{MODEL_ID}/drift")
    print(f"    curl 127.0.0.1:8000/models/{MODEL_ID}/performance")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(ap.parse_args().fast)
