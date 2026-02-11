from fastapi import APIRouter, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timezone
import asyncio
import json

from app.database import get_db, SessionLocal
from app.models.project import Project
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.models.dataset import Dataset
from app.schemas import TrainingConfigRequest, TrainingStatusResponse
from app.services.auth_service import get_user_from_header, get_current_user
from app.config import PROJECTS_DIR, SUPPORTED_MODELS
from app.ml.worker_registry import (
    get_worker,
    register_worker,
    unregister_worker,
    cleanup_dead_workers,
)

router = APIRouter(prefix="/projects/{project_id}/train", tags=["Training"])


def _get_project(project_id: int, authorization: Optional[str], db: Session):
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/configure", response_model=TrainingStatusResponse)
def configure_training(
    project_id: int,
    req: TrainingConfigRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    # Check datasets exist
    dataset_count = db.query(Dataset).filter(Dataset.project_id == project.id, Dataset.status == "ready").count()
    if dataset_count == 0:
        raise HTTPException(status_code=400, detail="No ready datasets. Please upload data first.")

    # Validate base model against supported models
    valid_model_ids = {m["model_id"] for m in SUPPORTED_MODELS}
    if req.base_model not in valid_model_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported base model: {req.base_model}. Choose from: {', '.join(valid_model_ids)}",
        )

    # Prevent configuring while training is active
    active_worker = get_worker(project.id)
    if active_worker and active_worker.is_alive:
        raise HTTPException(status_code=400, detail="Cannot reconfigure while training is active.")

    # Create or update model config
    config = ModelConfig(
        project_id=project.id,
        base_model=req.base_model,
        training_method=req.method,
        hyperparameters=req.model_dump(),
    )
    db.add(config)
    db.commit()
    db.refresh(config)

    # Create training run
    run = TrainingRun(
        project_id=project.id,
        model_config_id=config.id,
        total_epochs=req.epochs,
        status="configured",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return TrainingStatusResponse.model_validate(run)


@router.post("/start")
def start_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    # Clean up any dead workers first
    cleanup_dead_workers()

    # Check if training is already running for this project
    existing_worker = get_worker(project.id)
    if existing_worker and existing_worker.is_alive:
        raise HTTPException(status_code=400, detail="Training already in progress.")

    # Get latest training run
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project.id)
        .order_by(TrainingRun.id.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=400, detail="No training configured. Configure training first.")
    if run.status == "running":
        raise HTTPException(status_code=400, detail="Training already in progress.")

    # Get the model config for hyperparameters
    config = db.query(ModelConfig).filter(ModelConfig.id == run.model_config_id).first()
    if not config:
        raise HTTPException(status_code=400, detail="Training configuration not found.")

    # Update DB status
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    run.error_message = None  # Clear any previous error
    project.status = "training"
    db.commit()

    # Create and launch the real TrainingWorker
    hyperparams = config.hyperparameters or {}
    hyperparams["base_model"] = config.base_model
    hyperparams["method"] = config.training_method

    from app.ml.training_worker import TrainingWorker
    worker = TrainingWorker(
        project_id=project.id,
        run_id=run.id,
        config=hyperparams,
    )
    register_worker(project.id, worker)
    worker.start()

    return {"detail": "Training started", "run_id": run.id}


@router.post("/pause")
def pause_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    worker = get_worker(project.id)
    if not worker or not worker.is_alive:
        raise HTTPException(status_code=400, detail="No active training to pause.")

    worker.pause()

    # Update DB
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project.id, TrainingRun.status == "running")
        .order_by(TrainingRun.id.desc())
        .first()
    )
    if run:
        run.status = "paused"
        db.commit()

    return {"detail": "Training paused"}


@router.post("/resume")
def resume_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    worker = get_worker(project.id)
    if not worker or not worker.is_alive:
        raise HTTPException(status_code=400, detail="No active training to resume.")

    worker.resume()

    # Update DB
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project.id, TrainingRun.status == "paused")
        .order_by(TrainingRun.id.desc())
        .first()
    )
    if run:
        run.status = "running"
        db.commit()

    return {"detail": "Training resumed"}


@router.post("/stop")
def stop_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    worker = get_worker(project.id)
    if not worker or not worker.is_alive:
        raise HTTPException(status_code=400, detail="No active training to stop.")

    # Request graceful stop (saves checkpoint)
    worker.stop()

    # The worker thread handles DB finalization, but update project status immediately
    # for responsive UI
    project.status = "created"
    db.commit()

    # Unregister after a short delay to let the worker finalize
    # (the worker will be cleaned up by cleanup_dead_workers on next start)

    return {"detail": "Training stop requested. Saving checkpoint..."}


@router.get("/status", response_model=TrainingStatusResponse)
def training_status(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    # If there's a live worker, get real-time metrics from it
    worker = get_worker(project.id)
    if worker and worker.is_alive:
        status = worker.get_status()
        # Also get the run from DB for the ID and timestamps
        run = (
            db.query(TrainingRun)
            .filter(TrainingRun.project_id == project.id)
            .order_by(TrainingRun.id.desc())
            .first()
        )
        if run:
            return TrainingStatusResponse(
                id=run.id,
                status=status.get("status", run.status),
                current_loss=status.get("current_loss"),
                best_loss=status.get("best_loss"),
                validation_loss=status.get("validation_loss"),
                learning_rate_current=status.get("learning_rate_current"),
                current_epoch=status.get("current_epoch", 0),
                total_epochs=status.get("total_epochs", run.total_epochs),
                current_step=status.get("current_step", 0),
                total_steps=status.get("total_steps", 0),
                tokens_per_sec=status.get("tokens_per_sec", 0.0),
                error_message=status.get("error_message"),
                output_path=run.output_path,
                started_at=run.started_at,
                completed_at=run.completed_at,
            )

    # No live worker — read from DB
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project.id)
        .order_by(TrainingRun.id.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail="No training runs found.")
    return TrainingStatusResponse.model_validate(run)


@router.get("/history")
def training_history(
    project_id: int,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """List training runs with pagination and optional status filter."""
    project = _get_project(project_id, authorization, db)
    query = db.query(TrainingRun).filter(TrainingRun.project_id == project.id)
    if status:
        query = query.filter(TrainingRun.status == status)
    total = query.count()
    runs = query.order_by(TrainingRun.id.desc()).offset(max(0, offset)).limit(min(limit, 100)).all()
    return {
        "runs": [TrainingStatusResponse.model_validate(r) for r in runs],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.websocket("/ws")
async def training_ws(websocket: WebSocket, project_id: int):
    # Accept connection first — auth comes via first message (not query param)
    await websocket.accept()

    # Wait for auth message (timeout: 5 seconds)
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        auth_msg = __import__("json").loads(raw)
        token = auth_msg.get("token") if auth_msg.get("type") == "auth" else None
    except (asyncio.TimeoutError, Exception):
        await websocket.send_json({"type": "auth_error", "reason": "Authentication required — send {type: 'auth', token: '...'} as first message"})
        await websocket.close(code=4001, reason="Authentication timeout")
        return

    if not token:
        await websocket.send_json({"type": "auth_error", "reason": "Missing token"})
        await websocket.close(code=4001, reason="Missing token")
        return

    try:
        db = SessionLocal()
        user = get_current_user(db, token)
        project = db.query(Project).filter(
            Project.id == project_id, Project.user_id == user.id
        ).first()
        db.close()
        if not project:
            await websocket.send_json({"type": "auth_error", "reason": "Project not found"})
            await websocket.close(code=4003, reason="Project not found")
            return
    except Exception:
        await websocket.send_json({"type": "auth_error", "reason": "Invalid token"})
        await websocket.close(code=4001, reason="Invalid token")
        return

    try:
        while True:
            payload = {"status": "idle"}

            # Try to get live metrics from the worker
            worker = get_worker(project_id)
            if worker and worker.is_alive:
                status = worker.get_status()
                payload = {
                    "status": status.get("status", "running"),
                    "current_loss": status.get("current_loss"),
                    "best_loss": status.get("best_loss"),
                    "validation_loss": status.get("validation_loss"),
                    "current_epoch": status.get("current_epoch", 0),
                    "total_epochs": status.get("total_epochs", 0),
                    "current_step": status.get("current_step", 0),
                    "total_steps": status.get("total_steps", 0),
                    "tokens_per_sec": status.get("tokens_per_sec", 0.0),
                    "learning_rate_current": status.get("learning_rate_current"),
                    "error_message": status.get("error_message"),
                }

                # Calculate ETA
                if payload["current_step"] > 0 and payload["total_steps"] > 0:
                    db = SessionLocal()
                    try:
                        run = (
                            db.query(TrainingRun)
                            .filter(TrainingRun.project_id == project_id)
                            .order_by(TrainingRun.id.desc())
                            .first()
                        )
                        if run and run.started_at:
                            elapsed_seconds = (datetime.now(timezone.utc) - run.started_at).total_seconds()
                            secs_per_step = elapsed_seconds / payload["current_step"]
                            remaining_steps = payload["total_steps"] - payload["current_step"]
                            payload["eta_seconds"] = round(secs_per_step * remaining_steps)
                            payload["elapsed_seconds"] = round(elapsed_seconds)
                    finally:
                        db.close()
            else:
                # No active worker — check if there's a recent completed/error run
                db = SessionLocal()
                try:
                    run = (
                        db.query(TrainingRun)
                        .filter(TrainingRun.project_id == project_id)
                        .order_by(TrainingRun.id.desc())
                        .first()
                    )
                    if run and run.status in ("completed", "error", "stopped"):
                        payload = {
                            "status": run.status,
                            "current_loss": run.current_loss,
                            "best_loss": run.best_loss,
                            "current_epoch": run.current_epoch,
                            "total_epochs": run.total_epochs,
                            "current_step": run.current_step,
                            "total_steps": run.total_steps,
                            "tokens_per_sec": run.tokens_per_sec,
                            "error_message": run.error_message,
                        }
                finally:
                    db.close()

            # Always include live hardware stats
            try:
                from app.services.hardware_service import get_hardware_status
                payload["hardware"] = get_hardware_status()
            except Exception:
                payload["hardware"] = None

            await websocket.send_json(payload)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        # Graceful close on any unexpected error
        try:
            await websocket.close()
        except Exception:
            pass
