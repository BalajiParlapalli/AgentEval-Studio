"""SQLite database setup using SQLAlchemy."""
from sqlalchemy import create_engine, Column, String, Float, Integer, Text, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DB_PATH = os.getenv("DB_PATH", "results/agenteval.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer, default=0)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    dataset_id = Column(String, nullable=False)
    app_version = Column(String, default="v1")
    target_url = Column(String, nullable=False)
    status = Column(String, default="pending")   # pending | running | done | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    # Aggregate scores stored as JSON
    summary = Column(JSON, default=dict)


class EvalResult(Base):
    __tablename__ = "eval_results"
    id = Column(String, primary_key=True)
    run_id = Column(String, nullable=False, index=True)
    test_case_id = Column(String)
    input_text = Column(Text)
    expected_answer = Column(Text)
    actual_response = Column(Text)
    retrieved_contexts = Column(JSON, default=list)
    latency_ms = Column(Float, default=0.0)
    token_count = Column(Integer, default=0)
    # Scores
    keyword_coverage = Column(Float, default=0.0)
    rouge_l = Column(Float, default=0.0)
    faithfulness = Column(Float, default=0.0)
    answer_relevancy = Column(Float, default=0.0)
    context_recall = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    passed = Column(Integer, default=0)  # 0 or 1
    raw_scores = Column(JSON, default=dict)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
