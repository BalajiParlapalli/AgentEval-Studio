"""Pydantic v2 schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class TestCase(BaseModel):
    id: str
    input: str
    expected_answer: str
    expected_keywords: List[str] = []
    reference_context: str = ""
    app_version: str = "v1"
    category: str = "general"


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    cases: List[TestCase]


class DatasetOut(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    row_count: int


class RunCreate(BaseModel):
    name: str
    dataset_id: str
    app_version: str = "v1"
    target_url: str
    gemini_api_key: Optional[str] = None
    pass_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class RunOut(BaseModel):
    id: str
    name: str
    dataset_id: str
    app_version: str
    target_url: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    summary: dict


class ResultOut(BaseModel):
    id: str
    run_id: str
    test_case_id: str
    input_text: str
    expected_answer: str
    actual_response: str
    retrieved_contexts: List[Any]
    latency_ms: float
    token_count: int
    keyword_coverage: float
    rouge_l: float
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    overall_score: float
    passed: int
    raw_scores: dict
