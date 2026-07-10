import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://mlgpt:mlgpt@localhost:5432/mlgpt"
)
DRIFT_CHECK_INTERVAL_SECONDS = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "60"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
DRIFT_WINDOW_MINUTES = int(os.getenv("DRIFT_WINDOW_MINUTES", "10"))
MIN_ROWS_FOR_DRIFT = int(os.getenv("MIN_ROWS_FOR_DRIFT", "100"))

# check_same_thread=False lets the background drift scheduler and request
# threads share a SQLite connection (used for local/dev runs; ignored by Postgres)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
