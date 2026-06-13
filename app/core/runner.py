"""
Test runner: sends each test case to the target app, collects response + metadata.

Expected target API contract (your RAG Debate Arena):
  POST {target_url}
  Body: { "question": "...", "app_version": "..." }
  Response: {
      "answer": "...",
      "contexts": ["chunk1", "chunk2"],   # optional
      "token_count": 123                   # optional
  }

If your app uses a different schema, adjust _call_target() accordingly.
"""
import time
import uuid
import logging
from typing import Optional

import httpx

from app.core.metrics import (
    keyword_coverage,
    rouge_l_score,
    estimate_tokens,
    faithfulness_score,
    answer_relevancy_score,
    context_recall_score,
    compute_overall,
)
from app.models.schemas import TestCase

logger = logging.getLogger(__name__)


async def _call_target(
    target_url: str,
    case: TestCase,
    timeout: float = 30.0
) -> dict:
    """
    POST to the target RAG endpoint and return a normalised dict:
    { answer, contexts, token_count, latency_ms }
    """
    payload = {
        "question": case.input,
        "app_version": case.app_version,
        # RAG Debate Arena also accepts just "question"
    }
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(target_url, json=payload)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    try:
        data = r.json()
    except Exception:
        data = {}

    # Normalise response — support multiple schema shapes:
    # 1. Standard:      {"answer": "...", "contexts": [...]}
    # 2. RAG Debate:    {"verdict": "...", "pro": {"argument": ..., "evidence": [...]}, "anti": {...}}
    # 3. Generic:       {"response"/"output"/"result": "..."}
    if "verdict" in data:
        # RAG Debate Arena schema
        verdict = data.get("verdict", "")
        pro = data.get("pro", {})
        anti = data.get("anti", {})
        answer = (
            f"Verdict: {verdict}\n\n"
            f"Pro: {pro.get('argument', '')}\n\n"
            f"Anti: {anti.get('argument', '')}"
        ).strip() or r.text[:2000]
        pro_ev = pro.get("evidence", [])
        anti_ev = anti.get("evidence", [])
        if isinstance(pro_ev, str): pro_ev = [pro_ev]
        if isinstance(anti_ev, str): anti_ev = [anti_ev]
        contexts = pro_ev + anti_ev
    else:
        answer = (
            data.get("answer")
            or data.get("response")
            or data.get("output")
            or data.get("result")
            or r.text[:2000]
        )
        contexts = data.get("contexts") or data.get("retrieved_contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]

    token_count = data.get("token_count") or data.get("tokens") or estimate_tokens(answer)

    return {
        "answer": str(answer),
        "contexts": contexts,
        "token_count": int(token_count),
        "latency_ms": latency_ms,
        "status_code": r.status_code,
    }


async def run_single_case(
    case: TestCase,
    target_url: str,
    gemini_api_key: Optional[str] = None,
    pass_threshold: float = 0.6,
) -> dict:
    """Run one test case end-to-end and return a result dict."""
    result_id = str(uuid.uuid4())

    # 1. Call target
    try:
        raw = await _call_target(target_url, case)
    except Exception as e:
        logger.error("Target call failed for case %s: %s", case.id, e)
        raw = {
            "answer": f"[ERROR] {e}",
            "contexts": [],
            "token_count": 0,
            "latency_ms": 0.0,
            "status_code": 0,
        }

    answer = raw["answer"]
    contexts = raw["contexts"]

    # 2. Score
    kw = keyword_coverage(answer, case.expected_keywords)
    rl = rouge_l_score(answer, case.expected_answer)
    faith = faithfulness_score(answer, contexts, gemini_api_key)
    relevancy = answer_relevancy_score(answer, case.input, gemini_api_key)
    recall = context_recall_score(contexts, case.expected_answer, gemini_api_key)

    scores = {
        "keyword_coverage": kw,
        "rouge_l": rl,
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_recall": recall,
    }
    overall = compute_overall(scores)

    return {
        "id": result_id,
        "test_case_id": case.id,
        "input_text": case.input,
        "expected_answer": case.expected_answer,
        "actual_response": answer,
        "retrieved_contexts": contexts,
        "latency_ms": raw["latency_ms"],
        "token_count": raw["token_count"],
        "keyword_coverage": kw,
        "rouge_l": rl,
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_recall": recall,
        "overall_score": overall,
        "passed": int(overall >= pass_threshold),
        "raw_scores": scores,
    }
