"""Eval run endpoints."""
import uuid
import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.models.db import EvalRun, EvalResult, get_db
from app.models.schemas import RunCreate, RunOut, ResultOut
from app.core.evaluator import execute_run
from app.api.routes_datasets import load_cases

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("/", response_model=RunOut)
def create_run(
    body: RunCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    cases = load_cases(body.dataset_id)
    if not cases:
        raise HTTPException(400, "Dataset not found or empty.")

    run_id = str(uuid.uuid4())
    run = EvalRun(
        id=run_id,
        name=body.name,
        dataset_id=body.dataset_id,
        app_version=body.app_version,
        target_url=body.target_url,
        status="pending",
        summary={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Run in background so API returns immediately
    background_tasks.add_task(
        _run_wrapper,
        run_id=run_id,
        cases=cases,
        target_url=body.target_url,
        gemini_api_key=body.gemini_api_key,
        pass_threshold=body.pass_threshold,
    )
    return run


def _run_wrapper(**kwargs):
    """Sync wrapper to run async evaluator in background thread."""
    from app.models.db import SessionLocal
    db = SessionLocal()
    try:
        asyncio.run(execute_run(db=db, **kwargs))
    finally:
        db.close()


@router.get("/", response_model=List[RunOut])
def list_runs(db: Session = Depends(get_db)):
    return db.query(EvalRun).order_by(EvalRun.created_at.desc()).all()


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(404, "Run not found.")
    return run


@router.get("/{run_id}/results", response_model=List[ResultOut])
def get_results(run_id: str, db: Session = Depends(get_db)):
    results = db.query(EvalResult).filter(EvalResult.run_id == run_id).all()
    return results


@router.delete("/{run_id}")
def delete_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(404, "Not found.")
    db.query(EvalResult).filter(EvalResult.run_id == run_id).delete()
    db.delete(run)
    db.commit()
    return {"deleted": run_id}
