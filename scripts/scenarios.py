"""Labeled scenario suite — the acceptance test for MLGPT's core promise:
correctly naming WHY a model failed.

Each scenario drives a known failure through the real drift check + context
assembler and asserts the resulting `signature.label` matches ground truth. This
is the regression guard: if a change ever makes MLGPT misdiagnose a deploy bug as
a changed world (or vice versa), this fails loudly.

Runs fully in-process against a temp SQLite db — no server, no API key.

    .venv/bin/python scripts/scenarios.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# configure the backend BEFORE importing it, so config picks these up
_TMP = tempfile.mkdtemp(prefix="mlgpt_scen_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/scen.db"
os.environ["REFERENCE_DIR"] = f"{_TMP}/ref"
os.environ["PSI_THRESHOLD"] = "0.2"
os.environ["NULL_RATE_THRESHOLD"] = "0.25"
os.environ["DRIFT_WINDOW_MINUTES"] = "30"
os.environ["MIN_ROWS_FOR_DRIFT"] = "100"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.dirname(__file__))

import creditdata as C
from app.core.agent import _allowed_ref_ids, _validate, generate_explanation
from app.core.config import SessionLocal, engine
from app.core.context import assemble_incident_context
from app.core.drift import run_drift_check, save_reference
from app.models.db import Actual, Base, DriftReport, Event, Incident, Prediction

MODEL_ID = "credit-default"
N = 2000
DRIFT_AT = 900

# name, regime, has_deploy, expected signature label (None = no incident)
SCENARIOS = [
    ("healthy (no change)",            "baseline",     False, None),
    ("deploy flatten (value collapse)", "deploy_bug",   True,  "data_quality"),
    ("deploy scale (distribution shift)", "deploy_scale", True,  "deploy_or_pipeline"),
    ("null spike (broken upstream join)", "null_spike",   True,  "data_quality"),
    ("world change (population shift)", "pop_shift",     False, "world_change"),
]


def _wipe(db):
    for tbl in (Actual, Prediction, Event, DriftReport, Incident):
        db.query(tbl).delete()
    db.commit()
    ref = os.path.join(os.environ["REFERENCE_DIR"], f"{MODEL_ID}.parquet")
    if os.path.exists(ref):
        os.remove(ref)


def _populate(db, model, feats, reference, live_pool, regime, has_deploy):
    save_reference(MODEL_ID, reference)
    rng = np.random.default_rng(abs(hash(regime)) % 10000)
    Xb, yb, _ = C.regime_batch(model, feats, live_pool, "baseline", DRIFT_AT, rng)
    Xd, yd, _ = C.regime_batch(model, feats, live_pool, regime, N - DRIFT_AT, rng)
    X = pd.concat([Xb, Xd], ignore_index=True)
    y = np.concatenate([yb, yd])
    proba = model.predict_proba(X)[:, 1]

    now = datetime.now(timezone.utc)
    ts = ([now - timedelta(minutes=90) + timedelta(minutes=60) * i / DRIFT_AT
           for i in range(DRIFT_AT)] +
          [now - timedelta(minutes=30) + timedelta(minutes=30) * j / (N - DRIFT_AT)
           for j in range(N - DRIFT_AT)])

    preds = []
    for i in range(N):
        feat = {k: float(v) for k, v in X.iloc[i].items() if pd.notna(v)}  # NaN -> absent
        preds.append(Prediction(model_id=MODEL_ID, ts=ts[i], features=feat,
                                 prediction=float(proba[i]),
                                 model_version="2.3" if i >= DRIFT_AT else "2.2",
                                 ext_id=str(i)))
    db.add_all(preds)
    db.flush()
    db.add_all([Actual(prediction_id=preds[i].id, actual=float(y[i])) for i in range(N)])
    if has_deploy:
        db.add(Event(model_id=MODEL_ID, ts=now - timedelta(minutes=32), kind="deploy",
                     description="Deployed model v2.3"))
    db.commit()


def main() -> int:
    Base.metadata.create_all(engine)
    print("Training the real model once…")
    model, reference, m, feats, live_pool = C.train_model()
    print(f"  baseline AUC {m['auc']:.3f}\n")

    db = SessionLocal()
    passed = 0
    sample = None
    print(f"  {'scenario':<36}{'expected':<18}{'got':<18}{'conf':<8}{'cited':<7}result")
    print("  " + "-" * 96)
    for name, regime, has_deploy, expected in SCENARIOS:
        _wipe(db)
        _populate(db, model, feats, reference, live_pool, regime, has_deploy)
        run_drift_check()

        incident = db.execute(
            Incident.__table__.select().order_by(Incident.id.desc()).limit(1)
        ).first()
        cited = "-"
        if expected is None:
            got, conf = ("<no incident>", "-") if incident is None else ("<incident!>", "-")
            ok = incident is None
        elif incident is None:
            got, conf, ok = "<no incident!>", "-", False
        else:
            ctx = assemble_incident_context(db, incident.id)
            got = ctx["signature"]["label"]
            conf = ctx["signature"]["confidence"]
            # also verify the agent yields a valid, cited explanation (no key -> deterministic)
            expl = generate_explanation(ctx)
            valid, _ = _validate(expl, _allowed_ref_ids(ctx))
            cited = "ok" if valid else "BAD"
            ok = (got == expected) and valid
            if regime == "deploy_bug":
                sample = (name, expl)
        passed += ok
        print(f"  {name:<36}{str(expected):<18}{got:<18}{conf:<8}{cited:<7}{'PASS' if ok else 'FAIL'}")

    db.close()
    print(f"\n  {passed}/{len(SCENARIOS)} scenarios correctly diagnosed with a valid cited explanation.")
    if sample:
        name, e = sample
        print(f"\n  --- sample explanation ({name}, deterministic / no LLM key) ---")
        print(f"  summary        : {e['summary']}")
        print(f"  root_cause     : {e['root_cause']}")
        print(f"  confidence     : {e['confidence']}")
        print(f"  suggested_action: {e['suggested_action']}")
        print(f"  evidence       :")
        for ev in e["evidence"]:
            print(f"      - [{ev['type']}] ref={ev['ref_id']}  {ev['detail']}")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
