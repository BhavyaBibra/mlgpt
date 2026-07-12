import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://mlgpt:mlgpt@localhost:5432/mlgpt"
)
DRIFT_CHECK_INTERVAL_SECONDS = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "60"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
NULL_RATE_THRESHOLD = float(os.getenv("NULL_RATE_THRESHOLD", "0.25"))
# performance-decay trigger: catches concept/world drift that input-drift PSI
# can't see. Compares the recent labelled window against the earlier baseline.
PERF_ACC_DROP = float(os.getenv("PERF_ACC_DROP", "0.03"))
PERF_AUC_DROP = float(os.getenv("PERF_AUC_DROP", "0.03"))
PERF_MIN_LABELED = int(os.getenv("PERF_MIN_LABELED", "100"))
DRIFT_WINDOW_MINUTES = int(os.getenv("DRIFT_WINDOW_MINUTES", "10"))
MIN_ROWS_FOR_DRIFT = int(os.getenv("MIN_ROWS_FOR_DRIFT", "100"))

# check_same_thread=False lets the background drift scheduler and request
# threads share a SQLite connection (used for local/dev runs; ignored by Postgres)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)

# Root-cause agent LLM (Groq, OpenAI-compatible). Key supplied via env; when
# absent the agent falls back to a deterministic explanation from the signature.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Alerting. Without a webhook, alerts are logged (no-op) — demo works with no setup.
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
