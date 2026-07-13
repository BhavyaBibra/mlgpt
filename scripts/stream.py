"""Continuous live traffic — so the dashboard shows observability happening in
real time, not a static snapshot.

Streams real credit-default predictions + ground truth at a steady rate forever.
After --drift-after seconds it logs a deploy event and switches to the corrupted
regime, so you can watch — live — the feed keep flowing, the drift bars climb, an
incident open, and the cited explanation appear.

    python scripts/stream.py                 # healthy then drift at 45s
    python scripts/stream.py --drift-after 30 --rate 6

Needs the backend running on 127.0.0.1:8000.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
import creditdata as C
import mlgpt

MODEL_ID = "credit-default"


def main(drift_after: float, rate: float) -> None:
    print("→ Training the real model on UCI credit-default data…")
    model, reference, m, feats, live_pool = C.train_model()
    print(f"  baseline AUC {m['auc']:.3f}")

    mlgpt.init(api_url="http://127.0.0.1:8000", model_id=MODEL_ID, model_version="2.2")
    ref_path = "/tmp/mlgpt_stream_reference.csv"
    reference.to_csv(ref_path, index=False)
    print("  reference:", mlgpt.upload_reference(ref_path))

    rng = np.random.default_rng()
    start = time.time()
    i = 0
    deployed = False
    print(f"→ Streaming ~{rate}/s live (drift at t+{drift_after}s). Ctrl-C to stop.")
    while True:
        elapsed = time.time() - start
        drifting = elapsed >= drift_after
        if drifting and not deployed:
            mlgpt.log_event("Deployed model v2.3 (rewrote repayment-status "
                            "preprocessing)", kind="deploy")
            deployed = True
            print(f"\n  🚀 deploy v2.3 logged at t+{elapsed:.0f}s — drift begins\n")

        regime = "deploy_bug" if drifting else "baseline"
        n = int(rng.integers(2, 6))
        X, y, _ = C.regime_batch(model, feats, live_pool, regime, n, rng)
        proba = model.predict_proba(X)[:, 1]
        now = datetime.now(timezone.utc)
        acts = []
        for k in range(n):
            row = {c: float(v) for c, v in X.iloc[k].items() if v == v}  # drop NaN
            mlgpt.log_prediction(features=row, prediction=float(proba[k]),
                                 model_version="2.3" if drifting else "2.2",
                                 ext_id=str(i), ts=now)
            acts.append({"actual": float(y[k]), "ext_id": str(i)})
            i += 1
        mlgpt.flush()
        mlgpt.log_actuals(acts)
        if i % 40 < n:
            tag = " (DRIFTED)" if drifting else ""
            print(f"  t+{elapsed:4.0f}s  streamed {i} predictions{tag}")
        time.sleep(max(0.05, n / rate))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-after", type=float, default=45)
    ap.add_argument("--rate", type=float, default=5)
    try:
        a = ap.parse_args()
        main(a.drift_after, a.rate)
    except KeyboardInterrupt:
        print("\nstopped.")
