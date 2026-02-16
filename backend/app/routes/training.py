import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.project import Project
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.models.dataset import Dataset
from app.schemas import TrainingConfigRequest, TrainingStatusResponse, DetailWithIdResponse, DetailResponse, TrainingHistoryResponse, RunComparisonResponse, RunComparison, ConfigDiffEntry
from app.services.auth_service import get_user_from_header, get_current_user
from app.config import PROJECTS_DIR, MODEL_CATALOG
from app.dependencies import get_project_for_user
from app.ml.worker_registry import (
    get_worker,
    register_worker,
    unregister_worker,
    cleanup_dead_workers,
)

logger = logging.getLogger("meowllm.routes.training")

router = APIRouter(prefix="/projects/{project_id}/train", tags=["Training"])


@router.post("/configure", response_model=TrainingStatusResponse)
def configure_training(
    project_id: int,
    req: TrainingConfigRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)

    # Check datasets exist
    dataset_count = db.query(Dataset).filter(Dataset.project_id == project.id, Dataset.status == "ready").count()
    if dataset_count == 0:
        raise HTTPException(status_code=400, detail="No ready datasets. Please upload data first.")

    # Validate base model — accept any HuggingFace model ID (catalog or custom)
    if not req.base_model or "/" not in req.base_model:
        # Models must be in "org/name" format (e.g. "meta-llama/Llama-3.2-3B")
        catalog_ids = {m["model_id"] for m in MODEL_CATALOG}
        if req.base_model not in catalog_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model ID: '{req.base_model}'. Use 'org/model-name' format (e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0') or choose from the catalog.",
            )

    # Prevent configuring while training is active
    active_worker = get_worker(project.id)
    if active_worker and active_worker.is_alive:
        raise HTTPException(status_code=409, detail="Cannot reconfigure while training is active.")

    # Also check DB state for race condition safety
    running_run = db.query(TrainingRun).filter(
        TrainingRun.project_id == project.id,
        TrainingRun.status.in_(("running", "paused")),
    ).first()
    if running_run:
        raise HTTPException(status_code=409, detail="A training run is currently active. Stop it before reconfiguring.")

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


@router.post("/start", response_model=DetailWithIdResponse)
def start_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)

    # Clean up any dead workers first
    cleanup_dead_workers()

    # Check if training is already running for this project
    existing_worker = get_worker(project.id)
    if existing_worker and existing_worker.is_alive:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    # Get latest training run
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project.id)
        .order_by(TrainingRun.id.desc())
        .first()
    )
    if not run:
        raise HTTPException(status_code=400, detail="No training configured. Configure training first.")

    # Validate state machine: only configured/error/stopped/completed runs can be (re)started
    _STARTABLE_STATUSES = {"configured", "error", "stopped", "completed"}
    if run.status == "running":
        raise HTTPException(status_code=409, detail="Training already in progress.")
    if run.status == "paused":
        raise HTTPException(status_code=409, detail="Training is paused. Resume it instead of starting a new one.")
    if run.status not in _STARTABLE_STATUSES:
        raise HTTPException(status_code=400, detail=f"Cannot start training from status '{run.status}'.")

    # Get the model config for hyperparameters
    config = db.query(ModelConfig).filter(ModelConfig.id == run.model_config_id).first()
    if not config:
        raise HTTPException(status_code=400, detail="Training configuration not found.")

    # Verify datasets still exist
    dataset_count = db.query(Dataset).filter(
        Dataset.project_id == project.id,
        Dataset.status == "ready",
    ).count()
    if dataset_count == 0:
        raise HTTPException(status_code=400, detail="No ready datasets found. Upload data before training.")

    # Update DB status
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    run.completed_at = None  # Reset completion time for re-runs
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


@router.post("/pause", response_model=DetailResponse)
def pause_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)

    worker = get_worker(project.id)
    if not worker or not worker.is_alive:
        raise HTTPException(status_code=409, detail="No active training to pause.")

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


@router.post("/resume", response_model=DetailResponse)
def resume_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)

    worker = get_worker(project.id)
    if not worker or not worker.is_alive:
        raise HTTPException(status_code=409, detail="No active training to resume.")

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


@router.post("/stop", response_model=DetailResponse)
def stop_training(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = get_project_for_user(project_id, authorization, db)

    worker = get_worker(project.id)
    if not worker or not worker.is_alive:
        # Check DB for stale running status and clean it up
        run = (
            db.query(TrainingRun)
            .filter(
                TrainingRun.project_id == project.id,
                TrainingRun.status.in_(("running", "paused")),
            )
            .order_by(TrainingRun.id.desc())
            .first()
        )
        if run:
            run.status = "stopped"
            run.completed_at = datetime.now(timezone.utc)
            run.error_message = "Training process was no longer running"
            project.status = "created"
            db.commit()
            return {"detail": "Stale training run cleaned up."}
        raise HTTPException(status_code=409, detail="No active training to stop.")

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
    project = get_project_for_user(project_id, authorization, db)

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
                perplexity=status.get("perplexity"),
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


@router.get("/history", response_model=TrainingHistoryResponse)
def training_history(
    project_id: int,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """List training runs with pagination and optional status filter."""
    project = get_project_for_user(project_id, authorization, db)
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


@router.delete("/{run_id}", response_model=DetailResponse)
def delete_training_run(
    project_id: int,
    run_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Delete a specific training run. Cannot delete active (running/paused) runs."""
    project = get_project_for_user(project_id, authorization, db)

    run = db.query(TrainingRun).filter(
        TrainingRun.id == run_id,
        TrainingRun.project_id == project.id,
    ).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if run.status in ("running", "paused"):
        raise HTTPException(
            status_code=409,
            detail="Cannot delete an active training run. Stop it first.",
        )

    # Clean up associated checkpoint files if they exist
    checkpoint_dir = PROJECTS_DIR / str(project.id) / "checkpoints" / f"run_{run_id}"
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    # Also clean up the associated ModelConfig if no other runs reference it
    config_id = run.model_config_id
    db.delete(run)
    db.flush()

    if config_id:
        other_runs = db.query(TrainingRun).filter(
            TrainingRun.model_config_id == config_id,
        ).count()
        if other_runs == 0:
            config = db.query(ModelConfig).filter(ModelConfig.id == config_id).first()
            if config:
                db.delete(config)

    db.commit()
    return {"detail": f"Training run {run_id} deleted"}


@router.get("/compare", response_model=RunComparisonResponse)
def compare_runs(
    project_id: int,
    run_ids: str,  # comma-separated, e.g. "1,2"
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Compare two training runs side-by-side.

    Returns config diffs, metric summaries, and log histories for overlay charts.
    """
    project = get_project_for_user(project_id, authorization, db)

    try:
        ids = [int(x.strip()) for x in run_ids.split(",") if x.strip().isdigit()]
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid run_ids format. Use comma-separated integers.")
    if len(ids) != 2:
        raise HTTPException(status_code=400, detail="Provide exactly 2 run IDs (comma-separated).")
    if ids[0] == ids[1]:
        raise HTTPException(status_code=400, detail="Cannot compare a run with itself.")

    runs = []
    for rid in ids:
        run = db.query(TrainingRun).filter(
            TrainingRun.id == rid,
            TrainingRun.project_id == project.id,
        ).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Training run {rid} not found.")
        runs.append(run)

    comparisons = []
    for run in runs:
        config = db.query(ModelConfig).filter(ModelConfig.id == run.model_config_id).first()
        hyperparams = config.hyperparameters if config else {}

        comparisons.append({
            "run_id": run.id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "config": {
                "base_model": config.base_model if config else "unknown",
                "method": config.training_method if config else "unknown",
                **hyperparams,
            },
            "metrics": {
                "current_loss": run.current_loss,
                "best_loss": run.best_loss,
                "validation_loss": run.validation_loss,
                "perplexity": run.perplexity,
                "tokens_per_sec": run.tokens_per_sec,
                "current_step": run.current_step,
                "total_steps": run.total_steps,
                "current_epoch": run.current_epoch,
                "total_epochs": run.total_epochs,
            },
            "log_history": run.log_history[:2000] if run.log_history else [],
        })

    # Compute config diff
    config_a = comparisons[0]["config"]
    config_b = comparisons[1]["config"]
    all_keys = sorted(set(list(config_a.keys()) + list(config_b.keys())))
    config_diff = []
    for key in all_keys:
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        config_diff.append({
            "key": key,
            "run_a": val_a,
            "run_b": val_b,
            "different": val_a != val_b,
        })

    return {
        "runs": comparisons,
        "config_diff": config_diff,
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

    # Use a single DB session for the entire WS lifetime to avoid leaks
    ws_db = SessionLocal()
    _missed_sends = 0          # count of frames client couldn't keep up with
    _SEND_INTERVAL = 0.5       # base interval (seconds)
    _MAX_INTERVAL = 5.0        # backoff ceiling
    _MISS_THRESHOLD = 5        # disconnect after N consecutive missed sends
    _current_interval = _SEND_INTERVAL

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
                    try:
                        ws_db.expire_all()  # Refresh stale ORM objects
                        run = (
                            ws_db.query(TrainingRun)
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
                    except Exception:
                        ws_db.rollback()  # Reset session on error
            else:
                # No active worker — check if there's a recent completed/error run
                try:
                    ws_db.expire_all()  # Refresh stale ORM objects
                    run = (
                        ws_db.query(TrainingRun)
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
                except Exception:
                    ws_db.rollback()  # Reset session on error

            # Always include live hardware stats
            try:
                from app.services.hardware_service import get_hardware_status
                payload["hardware"] = get_hardware_status()
            except Exception:
                payload["hardware"] = None

            # ── Backpressure: try to send, back off if client is slow ──
            try:
                await asyncio.wait_for(
                    websocket.send_json(payload),
                    timeout=2.0,
                )
                # Successful send — reset miss counter, speed up
                _missed_sends = 0
                _current_interval = _SEND_INTERVAL
            except asyncio.TimeoutError:
                _missed_sends += 1
                # Exponential backoff (double interval, cap at MAX)
                _current_interval = min(_current_interval * 2, _MAX_INTERVAL)
                logger.warning(
                    "WS send timeout for project %d (miss #%d, next interval %.1fs)",
                    project_id, _missed_sends, _current_interval,
                )
                if _missed_sends >= _MISS_THRESHOLD:
                    logger.warning(
                        "Disconnecting slow WS client for project %d after %d missed sends",
                        project_id, _missed_sends,
                    )
                    await websocket.close(code=4008, reason="Client too slow")
                    break

            await asyncio.sleep(_current_interval)
    except WebSocketDisconnect:
        pass
    except Exception:
        # Graceful close on any unexpected error
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        ws_db.close()
