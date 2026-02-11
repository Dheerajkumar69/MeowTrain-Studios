from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File
from sqlalchemy.orm import Session
from typing import Optional
import os
import re
import uuid
from pathlib import Path

from app.database import get_db
from app.models.project import Project
from app.models.dataset import Dataset
from app.schemas import DatasetResponse, DatasetPreview
from app.services.auth_service import get_user_from_header
from app.config import PROJECTS_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from app.utils.text_extractor import extract_text

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["Datasets"])


def _sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters from uploaded filename."""
    # Take only the basename (prevent ../../../etc/passwd)
    name = os.path.basename(filename)
    # Remove any non-alphanumeric characters except . - _
    name = re.sub(r'[^\w.\-]', '_', name)
    return name or "unnamed"


def _get_project(project_id: int, authorization: Optional[str], db: Session):
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    # Sanitize filename and validate type
    safe_name = _sanitize_filename(file.filename or "unnamed")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Read file content
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    if file_size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum: {MAX_UPLOAD_SIZE_MB}MB")

    # Save file
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    dataset_dir = PROJECTS_DIR / str(project.id) / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_path = dataset_dir / stored_filename

    with open(file_path, "wb") as f:
        f.write(content)

    # Extract text and estimate tokens
    try:
        extracted = extract_text(file_path, ext)
        text = extracted["text"]
        # Rough token estimate: ~4 chars per token
        token_count = len(text) // 4
        chunk_count = max(1, token_count // 512)
        status = "ready"
    except Exception as e:
        # Clean up the file on extraction failure
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        token_count = 0
        chunk_count = 0
        status = "error"

    dataset = Dataset(
        project_id=project.id,
        filename=stored_filename,
        original_name=safe_name,
        file_type=ext,
        file_size=file_size,
        token_count=token_count,
        chunk_count=chunk_count,
        status=status,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return DatasetResponse.model_validate(dataset)


@router.get("/", response_model=list[DatasetResponse])
def list_datasets(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)
    datasets = db.query(Dataset).filter(Dataset.project_id == project.id).order_by(Dataset.uploaded_at.desc()).all()
    return [DatasetResponse.model_validate(d) for d in datasets]


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    project_id: int,
    dataset_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.project_id == project.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = PROJECTS_DIR / str(project.id) / "datasets" / dataset.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        extracted = extract_text(file_path, dataset.file_type)
        text = extracted["text"]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to extract text from file")

    # Create chunks
    chunk_size = 512 * 4  # ~512 tokens at ~4 chars/token
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        chunks.append({
            "text": chunk_text[:500] + ("..." if len(chunk_text) > 500 else ""),
            "token_count": len(chunk_text) // 4,
            "index": len(chunks),
        })
        if len(chunks) >= 20:  # Max 20 chunks in preview
            break

    return DatasetPreview(
        id=dataset.id,
        original_name=dataset.original_name,
        total_tokens=dataset.token_count,
        total_chunks=dataset.chunk_count,
        chunks=chunks,
    )


@router.delete("/{dataset_id}")
def delete_dataset(
    project_id: int,
    dataset_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.project_id == project.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete file
    file_path = PROJECTS_DIR / str(project.id) / "datasets" / dataset.filename
    if file_path.exists():
        file_path.unlink()

    db.delete(dataset)
    db.commit()
    return {"detail": "Dataset deleted"}
