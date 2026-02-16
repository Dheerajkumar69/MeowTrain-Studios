"""
Project backup / restore routes.

``GET  /api/projects/{id}/backup``   → download a .zip archive of the project
``POST /api/projects/import``        → upload a .zip archive to create a new project
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.models.prompt_template import PromptTemplate
from app.services.auth_service import get_user_from_header
from app.config import PROJECTS_DIR

logger = logging.getLogger("meowllm.routes.backup")

router = APIRouter(prefix="/projects", tags=["Backup"])


def _serialize_datetime(obj):
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@router.get("/{project_id}/backup")
def backup_project(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Export a project as a .zip archive containing metadata + datasets."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(
        Project.id == project_id,
        Project.user_id == user.id,
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Gather related data
    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()
    model_configs = db.query(ModelConfig).filter(ModelConfig.project_id == project_id).all()
    training_runs = db.query(TrainingRun).filter(TrainingRun.project_id == project_id).all()
    prompts = db.query(PromptTemplate).filter(PromptTemplate.project_id == project_id).all()

    # Build metadata JSON
    metadata = {
        "version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": project.name,
            "description": project.description,
            "intended_use": project.intended_use,
            "status": project.status,
        },
        "datasets": [
            {
                "id": ds.id,
                "filename": ds.filename,
                "original_name": ds.original_name,
                "file_type": ds.file_type,
                "file_size": ds.file_size,
                "token_count": ds.token_count,
                "chunk_count": ds.chunk_count,
                "status": ds.status,
            }
            for ds in datasets
        ],
        "model_configs": [
            {
                "id": mc.id,
                "base_model": mc.base_model,
                "training_method": mc.training_method,
                "hyperparameters": mc.hyperparameters,
                "status": mc.status,
                "created_at": mc.created_at,
            }
            for mc in model_configs
        ],
        "training_runs": [
            {
                "id": tr.id,
                "model_config_id": tr.model_config_id,
                "status": tr.status,
                "current_loss": tr.current_loss,
                "best_loss": tr.best_loss,
                "total_epochs": tr.total_epochs,
                "total_steps": tr.total_steps,
                "tokens_per_sec": tr.tokens_per_sec,
                "error_message": tr.error_message,
                "started_at": tr.started_at,
                "completed_at": tr.completed_at,
            }
            for tr in training_runs
        ],
        "prompt_templates": [
            {
                "name": pt.name,
                "system_prompt": pt.system_prompt,
                "user_prompt": pt.user_prompt,
                "temperature": pt.temperature,
            }
            for pt in prompts
        ],
    }

    # Create zip in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, default=_serialize_datetime, indent=2))

        # Include dataset files
        datasets_dir = PROJECTS_DIR / str(project_id) / "datasets"
        if datasets_dir.exists():
            for ds in datasets:
                file_path = datasets_dir / ds.filename
                if file_path.exists():
                    zf.write(file_path, f"datasets/{ds.filename}")

        # Include adapter files (trained models)
        adapters_dir = PROJECTS_DIR / str(project_id) / "adapters"
        if adapters_dir.exists():
            for fpath in adapters_dir.rglob("*"):
                if fpath.is_file():
                    arcname = f"adapters/{fpath.relative_to(adapters_dir)}"
                    zf.write(fpath, arcname)

    buf.seek(0)
    safe_name = project.name.replace(" ", "_").replace("/", "_")[:50]
    filename = f"meowllm_backup_{safe_name}_{project_id}.zip"

    logger.info("Exported project %d (%s) as backup archive", project_id, project.name)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/import")
def import_project(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Import a project from a .zip backup archive."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload must be a .zip file")

    # Read uploaded zip into memory (limit 500MB)
    content = file.file.read(500 * 1024 * 1024 + 1)
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Backup file too large (max 500MB)")

    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")

    # Read metadata
    try:
        raw_meta = zf.read("metadata.json")
        metadata = json.loads(raw_meta)
    except (KeyError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Missing or invalid metadata.json in archive")

    project_meta = metadata.get("project", {})
    if not project_meta.get("name"):
        raise HTTPException(status_code=400, detail="Project name missing in metadata")

    # Create new project
    new_project = Project(
        user_id=user.id,
        name=f"{project_meta['name']} (imported)",
        description=project_meta.get("description", ""),
        intended_use=project_meta.get("intended_use", "custom"),
        status="created",
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    project_dir = PROJECTS_DIR / str(new_project.id)
    datasets_dir = project_dir / "datasets"
    adapters_dir = project_dir / "adapters"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # ── Safe extraction helper (prevents zip-slip) ──
    def _safe_extract(zf_ref, arcname: str, dest_dir: Path) -> bool:
        """Extract a single file from the zip, ensuring the path stays within dest_dir."""
        target = (dest_dir / arcname).resolve()
        dest_resolved = dest_dir.resolve()
        if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
            logger.warning("Zip-slip attempt blocked: %s -> %s", arcname, target)
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf_ref.open(arcname) as src, open(target, "wb") as dst:
            dst.write(src.read())
        return True

    # Restore datasets
    dataset_count = 0
    for ds_meta in metadata.get("datasets", []):
        filename = ds_meta.get("filename", "")
        if not filename or ".." in filename or filename.startswith("/"):
            logger.warning("Skipping dataset with unsafe filename: %s", filename)
            continue
        arcname = f"datasets/{filename}"
        if arcname in zf.namelist():
            if _safe_extract(zf, arcname, project_dir):
                dataset_count += 1

        # Create DB record
        new_ds = Dataset(
            project_id=new_project.id,
            filename=filename,
            original_name=ds_meta.get("original_name", filename),
            file_type=ds_meta.get("file_type", "unknown"),
            file_size=ds_meta.get("file_size", 0),
            token_count=ds_meta.get("token_count", 0),
            chunk_count=ds_meta.get("chunk_count", 0),
            status=ds_meta.get("status", "ready"),
        )
        db.add(new_ds)

    # Restore prompt templates
    for pt_meta in metadata.get("prompt_templates", []):
        new_pt = PromptTemplate(
            project_id=new_project.id,
            name=pt_meta.get("name", "Imported Template"),
            system_prompt=pt_meta.get("system_prompt", ""),
            user_prompt=pt_meta.get("user_prompt", ""),
            temperature=pt_meta.get("temperature", 0.7),
        )
        db.add(new_pt)

    # Restore adapter files (safe extraction)
    adapter_files = [n for n in zf.namelist() if n.startswith("adapters/") and not n.endswith("/")]
    for arcname in adapter_files:
        _safe_extract(zf, arcname, project_dir)

    db.commit()
    zf.close()

    logger.info(
        "Imported project %d from backup: %d datasets, %d adapter files",
        new_project.id, dataset_count, len(adapter_files),
    )
    return {
        "detail": "Project imported successfully",
        "project_id": new_project.id,
        "datasets_restored": dataset_count,
    }
