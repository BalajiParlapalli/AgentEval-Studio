"""AgentEval Studio — FastAPI backend."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.db import init_db
from app.api.routes_datasets import router as datasets_router
from app.api.routes_runs import router as runs_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AgentEval Studio",
    description="Evaluation and observability dashboard for AI agents and RAG systems.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets_router, prefix="/api/v1")
app.include_router(runs_router, prefix="/api/v1")


@app.on_event("startup")
def on_startup():
    import os, logging
    init_db()
    key = os.getenv("GEMINI_API_KEY", "")
    logging.getLogger(__name__).info(
        "GEMINI_API_KEY: %s", "SET (len=%d)" % len(key) if key else "NOT SET — heuristic fallback active"
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "AgentEval Studio"}
