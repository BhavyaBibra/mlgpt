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

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
