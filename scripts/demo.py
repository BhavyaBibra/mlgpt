"""MLGPT demo — replays a synthetic credit-risk dataset as live traffic,
then injects drift so you can watch MLGPT catch and (in Phase 2) explain it.

Usage:
    python scripts/demo.py                 # ~3 minute demo
    python scripts/demo.py --fast          # ~40 second demo

Story the demo tells:
  t+0s    normal traffic flows (trained distribution)
  t+40%   "deploy v2.3" event logged
  t+45%   transaction_amount distribution shifts (the injected bug)
  ...     MLGPT's scheduled drift check flags it and opens an incident
"""
import argparse
import random
import time

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
import mlgpt

MODEL_ID = "credit-risk"
N_REFERENCE = 2000
N_LIVE = 1200


def make_row(drifted: bool = False) -> dict:
    """Synthetic credit application. Post-'deploy', transaction_amount
    shifts to a much larger distribution — the injected incident."""
    amount_mean, amount_sigma = (900, 1.1) if drifted else (300, 0.8)
    return {
        "transaction_amount": float(np.random.lognormal(np.log(amount_mean), amount_sigma)),
        "age": int(np.clip(np.random.normal(38, 12), 18, 85)),
        "income": float(np.random.lognormal(11, 0.5)),
        "credit_score": int(np.clip(np.random.normal(680, 70), 300, 850)),
        "num_accounts": int(np.clip(np.random.poisson(4), 0, 20)),
    }


def score(row: dict) -> float:
    """Stand-in model: risk score in [0,1]."""
    z = (
        0.5 * (row["transaction_amount"] / 1000)
        - 0.3 * (row["credit_score"] - 300) / 550
        - 0.2 * (row["income"] / 200000)
        + np.random.normal(0, 0.05)
    )
    return float(1 / (1 + np.exp(-4 * (z - 0.1))))


def main(fast: bool) -> None:
    delay = 0.005 if fast else 0.05

    print("→ Connecting to MLGPT at http://localhost:8000")
    mlgpt.init(api_url="http://localhost:8000", model_id=MODEL_ID, model_version="2.2")

    print(f"→ Generating reference data ({N_REFERENCE} rows) and uploading…")
    ref = pd.DataFrame([make_row() for _ in range(N_REFERENCE)])
    ref["__prediction__"] = [score(r) for r in ref.to_dict("records")]
    ref_path = "/tmp/mlgpt_demo_reference.csv"
    ref.to_csv(ref_path, index=False)
    print("  ", mlgpt.upload_reference(ref_path))

    deploy_at = int(N_LIVE * 0.40)
    drift_at = int(N_LIVE * 0.45)

    print(f"→ Streaming {N_LIVE} live predictions "
          f"(deploy event at #{deploy_at}, drift begins at #{drift_at})…")
    for i in range(N_LIVE):
        if i == deploy_at:
            evt = mlgpt.log_event("Deployed model v2.3 (new preprocessing for "
                                  "transaction_amount)", kind="deploy")
            print(f"\n  🚀 [{i}] deploy event logged (event id {evt.get('id')})\n")
        drifted = i >= drift_at
        row = make_row(drifted=drifted)
        mlgpt.log_prediction(features=row, prediction=score(row),
                             model_version="2.3" if i >= deploy_at else "2.2")
        if i % 100 == 0:
            print(f"  [{i}] streaming{' (DRIFTED distribution)' if drifted else ''}…")
        time.sleep(delay + random.uniform(0, delay))

    mlgpt.flush()
    print("\n✓ Demo traffic complete.")
    print("  Within one drift-check interval (default 60s) MLGPT will flag")
    print("  transaction_amount and open an incident.")
    print("  Check:  curl http://localhost:8000/incidents")
    print("  Drift:  curl http://localhost:8000/models/credit-risk/drift")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(ap.parse_args().fast)
