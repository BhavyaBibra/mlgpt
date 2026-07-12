"""creditsim — the "real world" that MLGPT's Option-B demo models.

This is the honest core of the demo: a REAL model (LogisticRegression — actual
learned weights) trained on a reference period, then scored against production
traffic under three regimes:

    baseline       world & inputs unchanged     -> accuracy holds
    deploy_bug     transaction_amount corrupted  -> input PSI spikes, accuracy drops
    concept_drift  the world changes gradually   -> inputs look normal, accuracy drops

The contrast between the last two IS MLGPT's core value. A broken deploy and a
genuinely changed world both crater accuracy, but only the deploy shows up as
input drift. Detecting the input shift is easy; telling the two apart is the
product.

Run standalone to see the mechanic with no database or backend:

    .venv/bin/python scripts/creditsim.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["credit_score", "income", "transaction_amount", "age", "num_accounts"]

# The "true" world: coefficients of the real (unknown to the model) process that
# decides who defaults. The model has to *learn* an approximation of these.
_WORLD_NORMAL = {
    "intercept": -1.8,
    "credit_score": -4.6,      # higher score -> less likely to default
    "income": -1.5,            # higher income -> less likely
    "transaction_amount": 3.8, # bigger recent spend -> more likely
    "age": -0.8,
    "num_accounts": 0.5,
}
# A genuinely changed world (recession-like): credit score stops protecting
# people and spending matters much more. INPUT distributions are unchanged —
# only the relationship to the outcome moved. This is concept drift.
_WORLD_SHIFTED = {
    "intercept": -0.8,
    "credit_score": -1.0,
    "income": -0.7,
    "transaction_amount": 5.0,
    "age": -0.4,
    "num_accounts": 1.0,
}


def sample_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n credit applications from the (stable) input distribution."""
    return pd.DataFrame({
        "credit_score": np.clip(rng.normal(680, 70, n), 300, 850),
        "income": rng.lognormal(11.0, 0.5, n),
        "transaction_amount": rng.lognormal(np.log(300), 0.8, n),
        "age": np.clip(rng.normal(38, 12, n), 18, 85),
        "num_accounts": np.clip(rng.poisson(4, n), 0, 20).astype(float),
    })


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw features onto a comparable scale for the true-process math."""
    return pd.DataFrame({
        "credit_score": (df["credit_score"] - 680) / 70,
        "income": (np.log(df["income"]) - 11.0) / 0.5,
        "transaction_amount": (np.log(df["transaction_amount"]) - np.log(300)) / 0.8,
        "age": (df["age"] - 38) / 12,
        "num_accounts": (df["num_accounts"] - 4) / 2,
    })


def true_default(df: pd.DataFrame, rng: np.random.Generator, world: dict) -> np.ndarray:
    """The real world's verdict: did each application actually default?

    This is ground truth — what `log_actual` eventually reports. The model never
    sees `world`; it only ever saw the NORMAL world during training.
    """
    z = _standardize(df)
    logodds = world["intercept"] + sum(world[f] * z[f] for f in FEATURES)
    prob = 1.0 / (1.0 + np.exp(-logodds))
    return (rng.random(len(df)) < prob).astype(int)


def train_model(seed: int = 7, n: int = 8000):
    """Train the real model on a reference period of the NORMAL world.

    Returns (fitted_pipeline, reference_dataframe, baseline_accuracy).
    """
    rng = np.random.default_rng(seed)
    X = sample_features(n, rng)
    y = true_default(X, rng, _WORLD_NORMAL)

    split = int(n * 0.75)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X.iloc[:split], y[:split])

    baseline_acc = float((model.predict(X.iloc[split:]) == y[split:]).mean())
    reference = X.iloc[:split].copy()
    reference["__prediction__"] = model.predict_proba(X.iloc[:split])[:, 1]
    return model, reference, baseline_acc


# ---------- production regimes ----------

def _predict_label(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def regime_batch(model, regime: str, n: int, rng: np.random.Generator,
                 concept_strength: float = 1.0):
    """Produce one batch of live traffic under a regime.

    Returns (features_as_seen_by_model, true_labels, accuracy).
    - baseline:      inputs and world unchanged.
    - deploy_bug:    a preprocessing bug multiplies transaction_amount ~3.5x
                     BEFORE it reaches the model; the true world is unchanged, so
                     the model's inputs are corrupted and its guesses go wrong.
    - concept_drift: inputs are drawn normally, but the true world is (partly)
                     the shifted one, blended by concept_strength in [0,1].
    """
    X_true = sample_features(n, rng)

    if regime == "concept_drift":
        world = {k: (1 - concept_strength) * _WORLD_NORMAL[k]
                    + concept_strength * _WORLD_SHIFTED[k] for k in _WORLD_NORMAL}
        y = true_default(X_true, rng, world)
        X_seen = X_true  # inputs look completely normal
    else:
        y = true_default(X_true, rng, _WORLD_NORMAL)
        X_seen = X_true.copy()
        if regime == "deploy_bug":
            X_seen["transaction_amount"] = X_seen["transaction_amount"] * 3.5

    acc = float((_predict_label(model, X_seen) == y).mean())
    return X_seen, y, acc


# ---------- tiny PSI (self-contained, no backend import) ----------

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.clip(np.histogram(reference, bins=edges)[0] / len(reference), 1e-4, None)
    cur_pct = np.clip(np.histogram(current, bins=edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _report():
    model, reference, baseline = train_model()
    coefs = dict(zip(FEATURES, model.named_steps["logisticregression"].coef_[0]))
    rng = np.random.default_rng(101)

    print("=" * 66)
    print("MLGPT Option-B core: a REAL trained model, and how it decays")
    print("=" * 66)
    print("\nLearned weights (the model's approximation of the true world):")
    for f, c in coefs.items():
        print(f"    {f:<20} {c:+.3f}")
    print(f"\nBaseline accuracy on held-out NORMAL traffic: {baseline*100:.1f}%")

    ref_amount = reference["transaction_amount"].to_numpy()
    print("\nProduction regimes (2000 rows each):")
    print(f"    {'regime':<16}{'accuracy':>10}{'txn_amount PSI':>18}   detected by")
    print("    " + "-" * 62)
    for regime in ["baseline", "deploy_bug", "concept_drift"]:
        X_seen, _y, acc = regime_batch(model, regime, 2000, rng)
        p = psi(ref_amount, X_seen["transaction_amount"].to_numpy())
        if regime == "baseline":
            detected = "-- (healthy)"
        elif p >= 0.2:
            detected = "INPUT DRIFT (PSI) + accuracy"
        else:
            detected = "ACCURACY ONLY (inputs look fine)"
        print(f"    {regime:<16}{acc*100:>9.1f}%{p:>18.3f}   {detected}")

    print("\nRead the last two rows: both crater accuracy, but only deploy_bug")
    print("shows up as input drift. That gap is exactly what the agent must")
    print("diagnose — and why performance monitoring can't be input-PSI alone.")


if __name__ == "__main__":
    _report()
