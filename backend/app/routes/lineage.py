"""
Model versioning & lineage routes.

Provides full traceability:  adapter → dataset(s) → hyperparameters → training metrics.

``GET /api/projects/{id}/lineage``              → full lineage graph for project
``GET /api/projects/{id}/lineage/runs/{run_id}`` → single run provenance
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.services.auth_service import get_user_from_header

logger = logging.getLogger("meowllm.routes.lineage")

router = APIRouter(prefix="/projects", tags=["Lineage"])


def _build_run_lineage(run: TrainingRun, model_config: ModelConfig, datasets: list) -> dict:
    """Build a lineage record for a single training run."""
    return {
        "run_id": run.id,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "model": {
            "config_id": model_config.id,
            "base_model": model_config.base_model,
            "training_method": model_config.training_method,
            "hyperparameters": model_config.hyperparameters or {},
        },
        "datasets": [
            {
                "id": ds.id,
                "original_name": ds.original_name,
                "file_type": ds.file_type,
                "token_count": ds.token_count,
                "chunk_count": ds.chunk_count,
                "status": ds.status,
            }
            for ds in datasets
        ],
        "metrics": {
            "current_loss": run.current_loss,
            "best_loss": run.best_loss,
            "validation_loss": run.validation_loss,
            "perplexity": run.perplexity,
            "tokens_per_sec": run.tokens_per_sec,
            "total_epochs": run.total_epochs,
            "total_steps": run.total_steps,
            "learning_rate_current": run.learning_rate_current,
        },
        "output_path": run.output_path,
        "checkpoint_path": run.checkpoint_path,
        "error_message": run.error_message,
    }


@router.get("/{project_id}/lineage")
def get_project_lineage(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Return the full lineage graph for a project.

    Includes every training run with its model config, datasets used,
    hyperparameters, and resulting metrics/output paths.
    """
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

    # Gather all training runs for this project
    runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.id.desc())
        .all()
    )

    # All datasets for the project (used by all runs as of the training date)
    all_datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()

    # All model configs
    config_map = {}
    configs = db.query(ModelConfig).filter(ModelConfig.project_id == project_id).all()
    for mc in configs:
        config_map[mc.id] = mc

    lineage_entries = []
    for run in runs:
        mc = config_map.get(run.model_config_id)
        if not mc:
            continue

        # Datasets available at training time = all datasets for the project
        # In a future enhancement, TrainingRun could store dataset IDs explicitly
        datasets_for_run = all_datasets

        lineage_entries.append(_build_run_lineage(run, mc, datasets_for_run))

    return {
        "project_id": project_id,
        "project_name": project.name,
        "total_runs": len(lineage_entries),
        "lineage": lineage_entries,
    }


@router.get("/{project_id}/lineage/runs/{run_id}")
def get_run_lineage(
    project_id: int,
    run_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Return provenance/lineage for a single training run."""
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

    run = db.query(TrainingRun).filter(
        TrainingRun.id == run_id,
        TrainingRun.project_id == project_id,
    ).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    mc = db.query(ModelConfig).filter(ModelConfig.id == run.model_config_id).first()
    if not mc:
        raise HTTPException(status_code=404, detail="Model config not found for this run")

    datasets = db.query(Dataset).filter(Dataset.project_id == project_id).all()

    return _build_run_lineage(run, mc, datasets)
