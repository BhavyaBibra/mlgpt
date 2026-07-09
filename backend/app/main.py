from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from app.api import ingest, query
from app.core.config import DRIFT_CHECK_INTERVAL_SECONDS, engine
from app.core.drift import run_drift_check
from app.models.db import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_drift_check, "interval", seconds=DRIFT_CHECK_INTERVAL_SECONDS)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(
    title="MLGPT",
    description="Ask your ML models why they're failing.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
