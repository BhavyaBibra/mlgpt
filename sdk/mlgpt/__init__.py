"""MLGPT SDK — instrument any model with one line.

    import mlgpt
    mlgpt.init(api_url="http://localhost:8000", model_id="fraud-v2")
    mlgpt.log_prediction(features={"amount": 120.5}, prediction=0.91)
    mlgpt.log_actual(prediction_id=1, actual=1)
    mlgpt.log_event("Deployed model v2.3", kind="deploy")

Predictions are buffered and flushed in batches so the caller pays
near-zero latency.
"""
import atexit
import threading
from datetime import datetime

import httpx

_config: dict = {}
_buffer: list[dict] = []
_lock = threading.Lock()
_FLUSH_SIZE = 200
_FLUSH_INTERVAL = 2.0  # seconds


def init(api_url: str = "http://localhost:8000", model_id: str = "default",
         model_version: str = "unknown") -> None:
    _config.update(api_url=api_url.rstrip("/"), model_id=model_id,
                   model_version=model_version)
    t = threading.Thread(target=_flusher, daemon=True)
    t.start()
    atexit.register(flush)


def log_prediction(features: dict, prediction: float,
                   model_id: str | None = None, model_version: str | None = None,
                   ts: datetime | None = None) -> None:
    item = {
        "model_id": model_id or _config["model_id"],
        "features": features,
        "prediction": float(prediction),
        "model_version": model_version or _config["model_version"],
    }
    if ts:
        item["ts"] = ts.isoformat()
    with _lock:
        _buffer.append(item)
        if len(_buffer) >= _FLUSH_SIZE:
            _flush_locked()


def log_actual(prediction_id: int, actual: float) -> None:
    httpx.post(f"{_config['api_url']}/ingest/actuals",
               json=[{"prediction_id": prediction_id, "actual": float(actual)}],
               timeout=10)


def log_event(description: str, kind: str = "note",
              model_id: str | None = None, ts: datetime | None = None) -> dict:
    payload = {
        "model_id": model_id or _config["model_id"],
        "description": description, "kind": kind,
    }
    if ts:
        payload["ts"] = ts.isoformat()
    r = httpx.post(f"{_config['api_url']}/ingest/events", json=payload, timeout=10)
    return r.json()


def upload_reference(csv_path: str, model_id: str | None = None) -> dict:
    mid = model_id or _config["model_id"]
    with open(csv_path, "rb") as f:
        r = httpx.post(f"{_config['api_url']}/models/{mid}/reference",
                       files={"file": ("reference.csv", f, "text/csv")}, timeout=60)
    return r.json()


def flush() -> None:
    with _lock:
        _flush_locked()


def _flush_locked() -> None:
    global _buffer
    if not _buffer:
        return
    batch, _buffer = _buffer, []
    try:
        httpx.post(f"{_config['api_url']}/ingest/predictions",
                   json={"predictions": batch}, timeout=30)
    except Exception:
        # drop rather than block the caller's prediction path
        pass


def _flusher() -> None:
    import time
    while True:
        time.sleep(_FLUSH_INTERVAL)
        flush()
