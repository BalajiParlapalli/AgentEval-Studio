"""Dataset CRUD endpoints."""
import uuid
import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.models.db import Dataset, get_db
from app.models.schemas import DatasetCreate, DatasetOut, TestCase

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _store_cases(dataset_id: str, cases: List[TestCase]):
    """Persist test cases to a JSON file."""
    import os
    os.makedirs("results", exist_ok=True)
    path = f"results/dataset_{dataset_id}.json"
    with open(path, "w") as f:
        json.dump([c.model_dump() for c in cases], f, indent=2)


def load_cases(dataset_id: str) -> List[TestCase]:
    path = f"results/dataset_{dataset_id}.json"
    try:
        with open(path) as f:
            raw = json.load(f)
        return [TestCase(**r) for r in raw]
    except FileNotFoundError:
        return []


@router.post("/", response_model=DatasetOut)
def create_dataset(body: DatasetCreate, db: Session = Depends(get_db)):
    ds_id = str(uuid.uuid4())
    ds = Dataset(
        id=ds_id,
        name=body.name,
        description=body.description,
        row_count=len(body.cases),
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    _store_cases(ds_id, body.cases)
    return ds


@router.post("/upload", response_model=DatasetOut)
async def upload_dataset(
    name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a CSV or JSON golden dataset file."""
    content = await file.read()
    filename = file.filename or ""

    if filename.endswith(".json"):
        raw = json.loads(content)
        if isinstance(raw, dict) and "cases" in raw:
            raw = raw["cases"]
    elif filename.endswith(".csv"):
        import csv, io
        reader = csv.DictReader(io.StringIO(content.decode()))
        raw = list(reader)
    else:
        raise HTTPException(400, "Only .json or .csv files supported.")

    cases = []
    for i, row in enumerate(raw):
        keywords = row.get("expected_keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        cases.append(TestCase(
            id=row.get("id", f"case_{i}"),
            input=row.get("input", row.get("question", "")),
            expected_answer=row.get("expected_answer", row.get("answer", "")),
            expected_keywords=keywords,
            reference_context=row.get("reference_context", ""),
            app_version=row.get("app_version", "v1"),
            category=row.get("category", "general"),
        ))

    ds_id = str(uuid.uuid4())
    ds = Dataset(id=ds_id, name=name, row_count=len(cases))
    db.add(ds)
    db.commit()
    db.refresh(ds)
    _store_cases(ds_id, cases)
    return ds


@router.get("/", response_model=List[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.get("/{dataset_id}/cases")
def get_cases(dataset_id: str):
    cases = load_cases(dataset_id)
    if not cases:
        raise HTTPException(404, "Dataset not found or empty.")
    return [c.model_dump() for c in cases]


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Not found.")
    db.delete(ds)
    db.commit()
    return {"deleted": dataset_id}
