from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File, Request
from sqlalchemy.orm import Session
from typing import Optional
import os
import re
import shutil
import uuid
from pathlib import Path

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import get_db
from app.models.project import Project
from app.models.dataset import Dataset
from app.schemas import DatasetResponse, DatasetPreview, TrainingPreviewResponse, DetailResponse, PaginatedDatasetsResponse
from app.services.auth_service import get_user_from_header
from app.config import PROJECTS_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, RATE_LIMIT_UPLOAD
from app.utils.text_extractor import extract_text
from app.dependencies import get_project_for_user

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except ImportError:
    def _count_tokens(text: str) -> int:
        return len(text) // 4  # fallback estimation

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["Datasets"])
limiter = Limiter(key_func=get_remote_address)


def _sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters from uploaded filename."""
    # Take only the basename (prevent ../../../etc/passwd)
    name = os.path.basename(filename)
    # Remove any non-alphanumeric characters except . - _
    name = re.sub(r'[^\w.\-]', '_', name)
    return name[:200] or "unnamed"


def _safe_resolve(base_dir: Path, relative: str) -> Path:
    """Resolve a path ensuring it stays within base_dir (prevents path traversal)."""
    resolved = (base_dir / relative).resolve()
    base_resolved = base_dir.resolve()
    if not str(resolved).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    return resolved


@router.post("/upload", response_model=DatasetResponse)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_dataset(
    request: Request,
    project_id: int,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)

    # Sanitize filename and validate type
    safe_name = _sanitize_filename(file.filename or "unnamed")
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Stream file to disk in chunks (never hold whole file in memory)
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    dataset_dir = PROJECTS_DIR / str(project.id) / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file_path = dataset_dir / stored_filename

    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    file_size = 0
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(64 * 1024)  # 64 KB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum: {MAX_UPLOAD_SIZE_MB}MB",
                    )
                f.write(chunk)
    except HTTPException:
        # Clean up partial file on size violation
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    if file_size == 0:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="File is empty.")

    # Extract text and estimate tokens
    try:
        extracted = extract_text(file_path, ext)
        text = extracted["text"]
        token_count = _count_tokens(text)
        chunk_count = max(1, token_count // 512)
        status = "ready"
    except Exception as e:
        # Clean up the file on extraction failure
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process file '{safe_name}': {e}"
        )

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


@router.get("/", response_model=PaginatedDatasetsResponse)
def list_datasets(
    project_id: int,
    page: int = 1,
    per_page: int = 50,
    search: Optional[str] = None,
    status: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)
    per_page = max(1, min(per_page, 200))
    page = max(1, page)

    query = db.query(Dataset).filter(Dataset.project_id == project.id)
    if search:
        query = query.filter(Dataset.original_name.ilike(f"%{search}%"))
    if status:
        query = query.filter(Dataset.status == status)

    total = query.count()
    datasets = (
        query
        .order_by(Dataset.uploaded_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )
    return {
        "items": [DatasetResponse.model_validate(d) for d in datasets],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(
    project_id: int,
    dataset_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.project_id == project.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_dir = PROJECTS_DIR / str(project.id) / "datasets"
    file_path = _safe_resolve(dataset_dir, dataset.filename)
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
            "token_count": _count_tokens(chunk_text),
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


@router.post("/preview-training", response_model=TrainingPreviewResponse)
def preview_training(
    project_id: int,
    base_model: str = "meta-llama/Llama-2-7b-hf",
    max_tokens: int = 512,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Analyse all ready datasets for a project and show:
    - Detected format per file
    - Sample training examples (raw → template-applied)
    - Token distribution stats
    """
    project = get_project_for_user(project_id, authorization, db)
    datasets_list = db.query(Dataset).filter(
        Dataset.project_id == project.id, Dataset.status == "ready"
    ).all()

    if not datasets_list:
        raise HTTPException(status_code=400, detail="No ready datasets found for this project")

    from app.ml.data_loader import detect_dataset_format, _extract_examples

    dataset_dir = PROJECTS_DIR / str(project.id) / "datasets"
    results = []
    all_examples = []

    for ds in datasets_list:
        file_path = dataset_dir / ds.filename
        if not file_path.exists():
            continue

        try:
            fmt_info = detect_dataset_format(file_path, ds.file_type)
            examples, fmt = _extract_examples(file_path, ds.file_type)

            # Build sample previews (up to 3 per file)
            sample_previews = []
            for ex in examples[:3]:
                preview = {
                    "raw": ex.get("text", "")[:500],
                    "has_messages": bool(ex.get("messages")),
                    "token_count": _count_tokens(ex.get("text", "")),
                }
                # Show chat template version if messages present
                if ex.get("messages"):
                    preview["messages"] = ex["messages"]
                    # Try to apply chat template
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(
                            base_model, trust_remote_code=False, use_fast=True,
                        )
                        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                            templated = tokenizer.apply_chat_template(
                                ex["messages"], tokenize=False, add_generation_prompt=False,
                            )
                            preview["templated"] = templated[:500]
                            preview["templated_tokens"] = _count_tokens(templated)
                    except Exception:
                        pass  # Template preview is best-effort
                sample_previews.append(preview)

            all_examples.extend(examples)

            results.append({
                "dataset_id": ds.id,
                "filename": ds.original_name,
                "format": fmt_info["format"],
                "format_description": fmt_info["description"],
                "sample_count": len(examples),
                "samples": sample_previews,
            })
        except Exception as e:
            results.append({
                "dataset_id": ds.id,
                "filename": ds.original_name,
                "format": "error",
                "format_description": f"Failed to analyse: {e}",
                "sample_count": 0,
                "samples": [],
            })

    # Aggregate stats
    token_counts = [_count_tokens(ex.get("text", "")) for ex in all_examples if ex.get("text")]
    has_instruction = sum(1 for ex in all_examples if ex.get("messages") or ex.get("instruction"))

    return {
        "datasets": results,
        "summary": {
            "total_examples": len(all_examples),
            "instruction_examples": has_instruction,
            "text_only_examples": len(all_examples) - has_instruction,
            "avg_tokens": round(sum(token_counts) / max(len(token_counts), 1)),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "total_tokens": sum(token_counts),
        },
    }


@router.delete("/{dataset_id}", response_model=DetailResponse)
def delete_dataset(
    project_id: int,
    dataset_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.project_id == project.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete file safely
    dataset_dir = PROJECTS_DIR / str(project.id) / "datasets"
    file_path = _safe_resolve(dataset_dir, dataset.filename)
    if file_path.exists():
        file_path.unlink()

    db.delete(dataset)
    db.commit()
    return {"detail": "Dataset deleted"}
