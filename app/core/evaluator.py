"""
Evaluator: orchestrates a full eval run over a dataset, saves results to DB.
"""
import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.db import EvalRun, EvalResult, Dataset
from app.models.schemas import TestCase
from app.core.runner import run_single_case

logger = logging.getLogger(__name__)


async def execute_run(
    run_id: str,
    cases: list[TestCase],
    target_url: str,
    gemini_api_key: Optional[str],
    pass_threshold: float,
    db: Session,
):
    """Execute all test cases for a run and persist results."""
    # Mark running
    run: EvalRun = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        logger.error("Run %s not found", run_id)
        return
    # Use env var as fallback (set via HF Space secrets)
    if not gemini_api_key:
        gemini_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")

    run.status = "running"
    db.commit()

    results = []
    for case in cases:
        try:
            result = await run_single_case(
                case=case,
                target_url=target_url,
                gemini_api_key=gemini_api_key,
                pass_threshold=pass_threshold,
            )
            result["run_id"] = run_id
            db_result = EvalResult(**result)
            db.add(db_result)
            db.commit()
            results.append(result)
            logger.info("Case %s done — overall=%.2f", case.id, result["overall_score"])
        except Exception as e:
            logger.error("Case %s failed: %s", case.id, e)

    # Aggregate summary
    if results:
        n = len(results)
        summary = {
            "total_cases": n,
            "passed": sum(r["passed"] for r in results),
            "pass_rate": round(sum(r["passed"] for r in results) / n, 4),
            "avg_overall": round(sum(r["overall_score"] for r in results) / n, 4),
            "avg_faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
            "avg_answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 4),
            "avg_context_recall": round(sum(r["context_recall"] for r in results) / n, 4),
            "avg_rouge_l": round(sum(r["rouge_l"] for r in results) / n, 4),
            "avg_keyword_coverage": round(sum(r["keyword_coverage"] for r in results) / n, 4),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 2),
            "avg_token_count": round(sum(r["token_count"] for r in results) / n, 1),
        }
    else:
        summary = {"total_cases": 0, "passed": 0, "pass_rate": 0.0}

    run.status = "done"
    run.completed_at = datetime.utcnow()
    run.summary = summary
    db.commit()
    logger.info("Run %s complete: %s", run_id, summary)
