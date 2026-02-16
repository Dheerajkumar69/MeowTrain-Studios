"""
Dataset augmentation API routes.

Provides endpoints for cleaning, deduplicating, and filtering
training datasets before training begins.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dataset import Dataset
from app.models.project import Project
from app.schemas import AugmentationResponse
from app.config import PROJECTS_DIR
from app.services.auth_service import get_user_from_header
from app.dependencies import get_project_for_user

logger = logging.getLogger("meowllm.routes.augmentation")

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["Augmentation"])


class AugmentRequest(BaseModel):
    enable_dedup: bool = Field(default=True, description="Remove near-duplicate examples")
    enable_clean: bool = Field(default=True, description="Clean text (HTML, encoding, whitespace)")
    enable_filter: bool = Field(default=True, description="Filter low-quality examples")
    dedup_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    min_length: int = Field(default=10, ge=1, le=10000)
    max_length: int = Field(default=100000, ge=100, le=1000000)
    strip_urls: bool = Field(default=False)
    strip_emails: bool = Field(default=False)
    preview_only: bool = Field(default=True, description="If true, only preview changes without applying")


@router.post("/augment", response_model=AugmentationResponse)
def augment_datasets(
    project_id: int,
    req: AugmentRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Run augmentation pipeline on project datasets.

    If preview_only=True (default), returns stats and sample changes
    without modifying the actual files. Set preview_only=False to apply.
    """
    project = get_project_for_user(project_id, authorization, db)

    # Get ready datasets
    datasets = db.query(Dataset).filter(
        Dataset.project_id == project.id,
        Dataset.status == "ready",
    ).all()

    if not datasets:
        raise HTTPException(status_code=400, detail="No ready datasets to augment.")

    # Load examples from all datasets
    from app.ml.data_loader import _extract_examples, _read_json_items

    dataset_dir = (PROJECTS_DIR / str(project.id) / "datasets").resolve()
    if not str(dataset_dir).startswith(str(PROJECTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project path")
    all_examples = []
    dataset_info = []

    for ds in datasets:
        file_path = (dataset_dir / ds.filename).resolve()
        if not str(file_path).startswith(str(dataset_dir)):
            logger.warning("Skipping dataset with path traversal: %s", ds.filename)
            continue
        if not file_path.exists():
            continue

        try:
            examples, fmt = _extract_examples(file_path, ds.file_type)
            dataset_info.append({
                "id": ds.id,
                "name": ds.original_name,
                "format": fmt,
                "example_count": len(examples),
            })
            all_examples.extend(examples)
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", ds.original_name, e)
            continue

    if not all_examples:
        raise HTTPException(status_code=400, detail="Could not extract examples from datasets.")

    # Run augmentation pipeline
    from app.ml.dataset_augmentation import run_augmentation_pipeline

    clean_options = {
        "strip_urls": req.strip_urls,
        "strip_emails": req.strip_emails,
    }

    result = run_augmentation_pipeline(
        examples=all_examples,
        enable_dedup=req.enable_dedup,
        enable_clean=req.enable_clean,
        enable_filter=req.enable_filter,
        dedup_threshold=req.dedup_threshold,
        min_length=req.min_length,
        max_length=req.max_length,
        clean_options=clean_options,
    )

    stats = result["stats"]

    # Build sample preview (show first 5 examples before/after)
    sample_before = all_examples[:5]
    sample_after = result["cleaned_examples"][:5]

    samples = []
    for i, (before, after) in enumerate(zip(sample_before, sample_after)):
        before_text = before.get("text", before.get("instruction", str(before)[:200]))
        after_text = after.get("text", after.get("instruction", str(after)[:200]))
        samples.append({
            "index": i,
            "before": before_text[:500] if isinstance(before_text, str) else str(before_text)[:500],
            "after": after_text[:500] if isinstance(after_text, str) else str(after_text)[:500],
            "changed": before_text != after_text,
        })

    response = {
        "preview_only": req.preview_only,
        "datasets": dataset_info,
        "stats": {
            "original_count": stats["original_count"],
            "final_count": stats["after_filter"],
            "duplicates_removed": stats["duplicates_removed"],
            "filtered_out": stats["filtered_out"],
            "filter_reasons": stats["filter_reasons"],
            "reduction_percent": round(
                (1 - stats["after_filter"] / max(stats["original_count"], 1)) * 100, 1
            ),
        },
        "samples": samples,
    }

    if not req.preview_only:
        # Apply: save cleaned data back as a new augmented dataset
        import json
        import time

        augmented_filename = f"augmented_{int(time.time())}.json"
        augmented_path = (dataset_dir / augmented_filename).resolve()
        if not str(augmented_path).startswith(str(dataset_dir)):
            raise HTTPException(status_code=400, detail="Invalid output path")

        with open(augmented_path, "w", encoding="utf-8") as f:
            json.dump(result["cleaned_examples"], f, ensure_ascii=False, indent=2)

        # Create DB record for augmented dataset
        new_ds = Dataset(
            project_id=project.id,
            filename=augmented_filename,
            original_name=f"Augmented ({stats['after_filter']} examples)",
            file_type="json",
            file_size=augmented_path.stat().st_size,
            token_count=0,
            chunk_count=stats["after_filter"],
            status="ready",
        )
        db.add(new_ds)
        db.commit()
        db.refresh(new_ds)

        response["created_dataset"] = {
            "id": new_ds.id,
            "filename": new_ds.original_name,
        }

    return response
