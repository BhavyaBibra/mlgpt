"""creditdata — real financial data for MLGPT's Option-B demo.

Loads the UCI "default of credit card clients" dataset (Taiwan, Yeh & Lien
2009): 30,000 real credit-card holders, 23 features, and a real
"defaulted next month" label. Auto-downloads via OpenML (no Kaggle login),
caches locally, and is a respected, citable credit-risk dataset.

A REAL model is trained on a reference period, then scored against production
traffic under three regimes:

    baseline       real holdout traffic            -> metrics hold
    deploy_bug     a preprocessing bug corrupts a   -> input drift + metrics drop
                   real feature at deploy time
    pop_shift      the incoming applicant mix moves -> metrics drop; distinguished
                   to a harder subpopulation           from a deploy by having no
                                                        correlated deploy event

Run standalone (no database/backend needed):

    .venv/bin/python scripts/creditdata.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# UCI column order for this dataset -> human-readable names.
_COLUMN_NAMES = [
    "credit_limit", "sex", "education", "marriage", "age",
    "pay_status_1", "pay_status_2", "pay_status_3",
    "pay_status_4", "pay_status_5", "pay_status_6",
    "bill_amt_1", "bill_amt_2", "bill_amt_3",
    "bill_amt_4", "bill_amt_5", "bill_amt_6",
    "pay_amt_1", "pay_amt_2", "pay_amt_3",
    "pay_amt_4", "pay_amt_5", "pay_amt_6",
]
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "credit_default.parquet")


def load_dataframe() -> pd.DataFrame:
    """Return the real dataset with readable columns + `defaulted` label, cached."""
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    from sklearn.datasets import fetch_openml
    ds = fetch_openml("default-of-credit-card-clients", version=1,
                      as_frame=True, parser="auto")
    X = ds.data.copy()
    X.columns = _COLUMN_NAMES
    X = X.apply(pd.to_numeric, errors="coerce")
    X["defaulted"] = ds.target.astype(int).to_numpy()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    X.to_parquet(CACHE)
    return X


def train_model(seed: int = 7):
    """Train the real model on a reference slice. Returns (model, reference, metrics)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    df = load_dataframe().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    feats = [c for c in df.columns if c != "defaulted"]
    n = len(df)
    ref_end = int(n * 0.6)          # reference / training period
    hold = slice(ref_end, int(n * 0.8))  # held-out "recent normal" traffic

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                            max_iter=250, random_state=seed)
    model.fit(df.loc[:ref_end - 1, feats], df.loc[:ref_end - 1, "defaulted"])

    Xh, yh = df.loc[hold, feats], df.loc[hold, "defaulted"]
    proba = model.predict_proba(Xh)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(yh, (proba >= 0.5).astype(int))),
        "auc": float(roc_auc_score(yh, proba)),
        "default_rate": float(yh.mean()),
        "majority_baseline": float(max(yh.mean(), 1 - yh.mean())),
    }
    reference = df.loc[:ref_end - 1, feats].copy()
    reference["__prediction__"] = model.predict_proba(reference[feats])[:, 1]
    # the live pool is the last 20% — real records the model never trained on
    live_pool = df.loc[int(n * 0.8):].reset_index(drop=True)
    return model, reference, metrics, feats, live_pool


# ---------- production regimes on real records ----------

def regime_batch(model, feats, live_pool, regime, n, rng):
    """One batch of live traffic under a regime. Returns (X_seen, y, metrics)."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    if regime == "pop_shift":
        # The applicant mix moves toward a harder subpopulation: recently late
        # payers with lower limits. Real records, real labels — the *population*
        # changed, not any single field's meaning.
        score = (live_pool["pay_status_1"].clip(lower=0) + 1) * \
                (live_pool["credit_limit"].max() - live_pool["credit_limit"] + 1)
        idx = live_pool.sample(n=n, replace=True, weights=score, random_state=rng.integers(1e9)).index
    else:
        idx = live_pool.sample(n=n, replace=True, random_state=rng.integers(1e9)).index

    batch = live_pool.loc[idx]
    X_seen, y = batch[feats].copy(), batch["defaulted"].to_numpy()

    if regime == "deploy_bug":
        # Bug shipped in v2.3: a rewrite of the repayment-status preprocessing
        # flattens every PAY_* code to 0 ("paid on time"). This block is the
        # model's strongest signal, so predictions collapse. (value collapse)
        for k in range(1, 7):
            X_seen[f"pay_status_{k}"] = 0
    elif regime == "deploy_scale":
        # A different shipped bug: an encoding change inverts the repayment-status
        # codes (late<->on-time), then shifts them. Distribution moves hard (high
        # PSI) with its spread intact (no collapse, no nulls), and because the
        # relationship is reversed the model's ranking breaks. Fingerprint:
        # deploy/pipeline corruption, not a data-quality null/collapse.
        for k in range(1, 7):
            X_seen[f"pay_status_{k}"] = 4 - X_seen[f"pay_status_{k}"]
    elif regime == "null_spike":
        # An upstream join breaks and the recent bill-statement fields arrive
        # empty for most rows — a null spike PSI can't see.
        mask = rng.random(len(X_seen)) < 0.7
        X_seen.loc[X_seen.index[mask], "bill_amt_1"] = np.nan

    proba = model.predict_proba(X_seen)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y, (proba >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y, proba)),
    }
    return X_seen, y, metrics


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), 1e-4, None)
    cur_pct = np.clip(np.histogram(current, bins=edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _report():
    model, reference, m, feats, live_pool = train_model()
    rng = np.random.default_rng(101)

    print("=" * 70)
    print("MLGPT Option-B: REAL model on REAL financial data (UCI credit default)")
    print("=" * 70)
    print(f"\nDataset: 30,000 real credit-card clients | default rate {m['default_rate']*100:.1f}%")
    print(f"Model: HistGradientBoosting trained on the reference period")
    print(f"Baseline on held-out real traffic:  accuracy {m['accuracy']*100:.1f}%  "
          f"AUC {m['auc']:.3f}  (majority-only = {m['majority_baseline']*100:.1f}%)")

    ref_pay = reference["pay_status_1"].to_numpy(dtype=float)
    base_auc = m["auc"]
    print("\nProduction regimes (3000 real rows each):")
    print(f"    {'regime':<12}{'AUC':>7}{'accuracy':>10}{'max PSI':>9}   fingerprint")
    print("    " + "-" * 82)
    for regime in ["baseline", "deploy_bug", "pop_shift"]:
        X_seen, _y, mm = regime_batch(model, feats, live_pool, regime, 3000, rng)
        max_psi = max(psi(reference[f].to_numpy(dtype=float),
                          X_seen[f].to_numpy(dtype=float)) for f in feats)
        auc_drop = base_auc - mm["auc"]
        if regime == "baseline":
            fp = "-- (healthy)"
        elif max_psi >= 1.0 and auc_drop >= 0.03:
            fp = "big feature drift + AUC collapse -> DEPLOY/PIPELINE bug"
        elif auc_drop < 0.02:
            fp = "AUC holds, accuracy fell, mild drift -> POPULATION/CONCEPT shift"
        else:
            fp = "mixed -> needs the event timeline to disambiguate"
        print(f"    {regime:<12}{mm['auc']:>7.3f}{mm['accuracy']*100:>9.1f}%{max_psi:>9.2f}   {fp}")

    print("\nSame symptom (the model is failing), two different fingerprints:")
    print("  - DEPLOY bug: a feature's PSI explodes and AUC collapses together.")
    print("  - POPULATION/CONCEPT: ranking (AUC) survives, accuracy/calibration")
    print("    erodes, no single feature explodes.")
    print("Correlating with the deploy timeline is the tie-breaker the agent uses.")


if __name__ == "__main__":
    _report()
