import os
import re
import threading
import logging
import time
import shutil
import zipfile
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.models.project import Project
from app.models.background_task import BackgroundTask
from app.schemas import (
    ModelInfo, ModelStatusResponse, DownloadStartResponse,
    DownloadProgressResponse, DetailResponse, GGUFExportResponse, GGUFStatusResponse,
)
from app.config import MODEL_CATALOG, MODEL_CACHE_DIR, PROJECTS_DIR
from app.services.hardware_service import get_hardware_status
from app.services.auth_service import get_user_from_header

logger = logging.getLogger("meowllm.models")

router = APIRouter(prefix="/models", tags=["Models"])

# ── In-memory locks (for thread safety of DB writes only) ────────────
_download_lock = threading.Lock()
_gguf_lock = threading.Lock()
_DOWNLOAD_TTL = 3600  # 1 hour — auto-cleanup completed/errored downloads

# ── Validation ──────────────────────────────────────────────────────
_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in filenames."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)[:200]


# ═══════════════════════════════════════════════════════════════════
#  DB-backed task helpers (replace the old in-memory dicts)
# ═══════════════════════════════════════════════════════════════════

def _get_or_create_task(
    db: Session,
    task_type: str,
    task_key: str,
    *,
    initial_status: str = "running",
    initial_message: str = "",
    metadata: dict | None = None,
) -> BackgroundTask:
    """Get an existing active task or create a new one."""
    task = (
        db.query(BackgroundTask)
        .filter(
            BackgroundTask.task_type == task_type,
            BackgroundTask.task_key == task_key,
            BackgroundTask.status.in_(("running", "queued")),
        )
        .first()
    )
    if task:
        return task

    task = BackgroundTask(
        task_type=task_type,
        task_key=task_key,
        status=initial_status,
        progress=0.0,
        message=initial_message,
        metadata_=metadata or {},
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _update_task(task_type: str, task_key: str, **fields) -> None:
    """Update a background task from a worker thread (gets its own session)."""
    db = SessionLocal()
    try:
        task = (
            db.query(BackgroundTask)
            .filter(
                BackgroundTask.task_type == task_type,
                BackgroundTask.task_key == task_key,
                BackgroundTask.status.in_(("running", "queued")),
            )
            .first()
        )
        if task:
            for k, v in fields.items():
                setattr(task, k, v)
            task.updated_at = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        logger.warning("Failed to update %s task '%s': %s", task_type, task_key, e)
        db.rollback()
    finally:
        db.close()


def _finish_task(task_type: str, task_key: str, *, status: str, message: str = "", error: str | None = None, metadata: dict | None = None) -> None:
    """Mark a background task as completed/error from a worker thread."""
    db = SessionLocal()
    try:
        task = (
            db.query(BackgroundTask)
            .filter(
                BackgroundTask.task_type == task_type,
                BackgroundTask.task_key == task_key,
                BackgroundTask.status.in_(("running", "queued")),
            )
            .first()
        )
        if task:
            task.status = status
            task.progress = 100.0 if status == "completed" else 0.0
            task.message = message
            task.error = error
            if metadata:
                task.metadata_ = {**(task.metadata_ or {}), **metadata}
            task.updated_at = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        logger.warning("Failed to finish %s task '%s': %s", task_type, task_key, e)
        db.rollback()
    finally:
        db.close()


def _get_task_status(db: Session, task_type: str, task_key: str) -> BackgroundTask | None:
    """Get the latest task for a given type/key (active first, then most recent)."""
    # Look for active task first
    task = (
        db.query(BackgroundTask)
        .filter(
            BackgroundTask.task_type == task_type,
            BackgroundTask.task_key == task_key,
            BackgroundTask.status.in_(("running", "queued")),
        )
        .first()
    )
    if task:
        return task

    # Fall back to latest completed/errored (within TTL)
    return (
        db.query(BackgroundTask)
        .filter(
            BackgroundTask.task_type == task_type,
            BackgroundTask.task_key == task_key,
        )
        .order_by(BackgroundTask.updated_at.desc())
        .first()
    )


def _cleanup_stale_tasks(db: Session) -> None:
    """Remove completed/errored tasks older than TTL."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_DOWNLOAD_TTL)
    db.query(BackgroundTask).filter(
        BackgroundTask.status.in_(("completed", "error", "cancelled", "interrupted")),
        BackgroundTask.updated_at < cutoff,
    ).delete(synchronize_session=False)
    db.commit()


def _check_compatibility(model: dict, hw: dict) -> str:
    """Check if model is compatible with current hardware."""
    ram_available = hw.get("ram_available_gb", 0)
    gpu_vram = hw.get("gpu_vram_available_gb", 0) or 0

    if hw.get("gpu_available") and gpu_vram >= model["vram_required_gb"]:
        return "compatible"
    elif ram_available >= model["ram_required_gb"]:
        if hw.get("gpu_available"):
            return "compatible"
        else:
            return "may_be_slow"
    elif ram_available >= model["ram_required_gb"] * 0.7:
        return "may_be_slow"
    else:
        return "too_large"


def _is_model_cached(model_id: str) -> bool:
    """Check if a model is already cached locally."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = "models--" + model_id.replace("/", "--")
    if os.path.isdir(os.path.join(hf_cache, model_dir_name)):
        snapshots_dir = os.path.join(hf_cache, model_dir_name, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                latest_snapshot = os.path.join(snapshots_dir, snapshots[-1])
                if os.listdir(latest_snapshot):
                    return True
    if (MODEL_CACHE_DIR / model_dir_name).is_dir():
        return True
    return False


def _download_model_thread(model_id: str, model_name: str):
    """Background thread that downloads a model from HuggingFace Hub.
    
    Cooperatively checks the DB for cancellation every few seconds
    so that DELETE /{model_id}/download actually stops the download.
    """
    try:
        with _download_lock:
            _update_task("download", model_id,
                         status="running",
                         progress=0.0,
                         message=f"Starting download of {model_name}...")

        from app.utils.lazy_imports import huggingface_hub as _hf_hub
        snapshot_download = _hf_hub().snapshot_download

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        _update_task("download", model_id,
                     progress=5.0,
                     message=f"Downloading {model_name} from HuggingFace...")

        logger.info("Starting download: %s", model_id)

        # Use a threading.Event that we check inside a tqdm callback
        cancel_event = threading.Event()

        def _check_cancelled():
            """Poll the DB to see if the user requested cancellation."""
            db = SessionLocal()
            try:
                task = (
                    db.query(BackgroundTask)
                    .filter(
                        BackgroundTask.task_type == "download",
                        BackgroundTask.task_key == model_id,
                    )
                    .order_by(BackgroundTask.updated_at.desc())
                    .first()
                )
                return task is not None and task.status == "cancelled"
            except Exception:
                return False
            finally:
                db.close()

        # Start a watchdog thread that periodically checks for cancellation
        def _cancellation_watchdog():
            while not cancel_event.is_set():
                if _check_cancelled():
                    cancel_event.set()
                    logger.info("Download cancellation detected for %s", model_id)
                    break
                cancel_event.wait(3)  # check every 3 seconds

        watchdog = threading.Thread(target=_cancellation_watchdog, daemon=True,
                                    name=f"cancel-watch-{model_id.replace('/', '-')}")
        watchdog.start()

        local_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            ignore_patterns=[
                "*.gguf", "*.bin", "*.msgpack",
                "consolidated.*", "original/**",
            ],
        )

        # Check one final time — download might have completed before cancel registered
        if cancel_event.is_set():
            _finish_task("download", model_id,
                         status="cancelled",
                         message=f"Download of {model_name} was cancelled.")
            logger.info("Download cancelled (post-completion): %s", model_id)
            return

        cancel_event.set()  # stop watchdog

        _finish_task("download", model_id,
                     status="completed",
                     message=f"{model_name} downloaded successfully!",
                     metadata={"local_path": local_path})

        logger.info("Download completed: %s -> %s", model_id, local_path)

    except Exception as e:
        error_msg = str(e)

        if "401" in error_msg or "Unauthorized" in error_msg:
            error_msg = (
                f"Access denied for {model_name}. This is a gated model that requires "
                f"a HuggingFace access token. Set the HF_TOKEN environment variable "
                f"and ensure you've accepted the model's license on huggingface.co."
            )
        elif "404" in error_msg:
            error_msg = f"Model {model_id} not found on HuggingFace Hub."
        elif "disk" in error_msg.lower() or "space" in error_msg.lower():
            error_msg = f"Not enough disk space to download {model_name}."

        logger.error("Download failed for %s: %s", model_id, error_msg)

        _finish_task("download", model_id,
                     status="error",
                     message=error_msg,
                     error=error_msg)


@router.get("/", response_model=list[ModelInfo])
def list_models(db: Session = Depends(get_db)):
    hw = get_hardware_status()
    models = []
    for m in MODEL_CATALOG:
        models.append(ModelInfo(
            model_id=m["model_id"],
            name=m["name"],
            description=m["description"],
            parameters=m["parameters"],
            size_gb=m["size_gb"],
            ram_required_gb=m["ram_required_gb"],
            vram_required_gb=m["vram_required_gb"],
            recommended_hardware=m["recommended_hardware"],
            estimated_train_minutes=m["estimated_train_minutes"],
            icon=m["icon"],
            is_cached=_is_model_cached(m["model_id"]),
            compatibility=_check_compatibility(m, hw),
        ))
    return models


@router.post("/custom/lookup", response_model=ModelInfo)
def lookup_custom_model(
    model_id: str = "",
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Look up a custom HuggingFace model by ID and return its info."""
    # Auth required to prevent anonymous HF API abuse
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    model_id = (model_id or "").strip()
    if not model_id or not _MODEL_ID_RE.match(model_id):
        raise HTTPException(
            status_code=400,
            detail="Model ID must be in 'org/name' format using only alphanumeric, dots, hyphens, and underscores "
                   "(e.g. 'meta-llama/Llama-3.2-3B')."
        )

    try:
        from app.utils.lazy_imports import huggingface_hub as _hf_hub
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        info = _hf_hub().model_info(model_id, token=hf_token)
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied for '{model_id}'. This may be a gated model. "
                       f"Set HF_TOKEN and accept the license on huggingface.co."
            )
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found on HuggingFace Hub."
        )

    # Estimate size from model info
    size_bytes = sum(s.size for s in (info.siblings or []) if s.size)
    size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0.0

    # Try to determine parameter count from tags or config
    params = "Unknown"
    for tag in (info.tags or []):
        if tag.startswith("params:"):
            params = tag.split(":")[1]
            break

    # Estimate resource requirements based on size
    ram_required = max(4, int(size_gb * 2.5))
    vram_required = max(2, int(size_gb * 1.2))

    hw = get_hardware_status()
    model_dict = {
        "model_id": model_id,
        "name": model_id.split("/")[-1],
        "description": f"Custom model from HuggingFace Hub. Pipeline: {info.pipeline_tag or 'unknown'}.",
        "parameters": params,
        "size_gb": size_gb,
        "ram_required_gb": ram_required,
        "vram_required_gb": vram_required,
        "recommended_hardware": f"{vram_required}GB+ VRAM or {ram_required}GB+ RAM",
        "estimated_train_minutes": max(10, int(size_gb * 5)),
        "icon": "🤗",
        "is_cached": _is_model_cached(model_id),
        "compatibility": _check_compatibility(
            {"vram_required_gb": vram_required, "ram_required_gb": ram_required}, hw
        ),
    }

    return ModelInfo(**model_dict)


# ── Export & GGUF routes (MUST come before /{model_id:path} routes) ──


@router.get("/export/{project_id}")
def export_trained_model(
    project_id: int,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Export the trained model/adapter for a project as a downloadable zip."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check for adapters or checkpoints (with path traversal protection)
    project_dir = (PROJECTS_DIR / str(project_id)).resolve()
    projects_base = PROJECTS_DIR.resolve()
    if not str(project_dir).startswith(str(projects_base)):
        raise HTTPException(status_code=400, detail="Invalid project path")

    adapters_dir = project_dir / "adapters"
    checkpoints_dir = project_dir / "checkpoints"

    # Find the best source to export
    export_dir = None
    if adapters_dir.exists() and any(adapters_dir.iterdir()):
        export_dir = adapters_dir
    elif checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        export_dir = checkpoints_dir
    else:
        raise HTTPException(status_code=404, detail="No trained model found. Complete training first.")

    # Create zip archive
    zip_filename = f"meowllm-{_sanitize_filename(project.name)}-model.zip"
    zip_path = project_dir / zip_filename

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            total_size = 0
            _ZIP_MAX_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB cap
            for root, _dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = Path(root) / file
                    total_size += file_path.stat().st_size
                    if total_size > _ZIP_MAX_SIZE:
                        try:
                            zip_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail="Model export exceeds 10 GB limit. Use GGUF export for large models."
                        )
                    arcname = file_path.relative_to(export_dir)
                    zf.write(file_path, arcname)

        # Schedule zip file cleanup after download completes
        def _cleanup_zip():
            try:
                zip_path.unlink(missing_ok=True)
                logger.info("Cleaned up export zip: %s", zip_path)
            except Exception as e:
                logger.warning("Failed to clean up export zip %s: %s", zip_path, e)

        background_tasks.add_task(_cleanup_zip)

        return FileResponse(
            path=str(zip_path),
            filename=zip_filename,
            media_type="application/zip",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create model export: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create model export")


@router.post("/export/{project_id}/gguf", response_model=GGUFExportResponse)
def export_gguf(
    project_id: int,
    quantization: str = "Q8_0",
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Convert trained model to GGUF format for LM Studio."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    valid_quants = {"f16", "Q8_0", "Q4_K_M"}
    if quantization not in valid_quants:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quantization: {quantization}. Choose from: {', '.join(valid_quants)}"
        )

    task_key = str(project_id)

    # Check for existing active conversion
    existing = _get_task_status(db, "gguf", task_key)
    if existing and existing.status in ("running", "queued"):
        return {"detail": "GGUF conversion already in progress", "status": existing.status}

    # Create DB task record
    task = _get_or_create_task(
        db, "gguf", task_key,
        initial_status="running",
        initial_message="Starting GGUF conversion...",
        metadata={"quantization": quantization, "project_name": project.name},
    )

    # Mutable dict for real-time progress (gguf_converter writes to this)
    status_dict = {"step": "queued", "progress": 0, "message": "Starting GGUF conversion..."}

    def _convert():
        try:
            from app.services.gguf_converter import export_project_gguf
            export_project_gguf(project_id, quantization, status_dict)

            # Persist final state to DB
            _finish_task("gguf", task_key,
                         status="completed",
                         message=status_dict.get("message", "GGUF conversion complete"),
                         metadata={
                             "gguf_filename": status_dict.get("gguf_filename"),
                             "gguf_size_mb": status_dict.get("gguf_size_mb"),
                         })
        except Exception as e:
            logger.error("GGUF conversion failed: %s", e)
            _finish_task("gguf", task_key,
                         status="error",
                         message=f"GGUF conversion failed: {e}",
                         error=str(e))

    thread = threading.Thread(target=_convert, daemon=True, name=f"gguf-{project_id}")
    thread.start()

    return {
        "detail": f"GGUF conversion started (quantization: {quantization})",
        "status": "converting",
    }


@router.get("/export/{project_id}/gguf/status", response_model=GGUFStatusResponse)
def gguf_status(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Check the status of a GGUF conversion."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    task_key = str(project_id)

    # Check DB for task
    task = _get_task_status(db, "gguf", task_key)
    if task:
        meta = task.metadata_ or {}
        if task.status == "completed":
            return {
                "step": "completed",
                "progress": 100,
                "message": task.message or "GGUF file ready for download",
                "gguf_filename": meta.get("gguf_filename"),
                "gguf_size_mb": meta.get("gguf_size_mb"),
            }
        elif task.status in ("running", "queued"):
            return {
                "step": "converting",
                "progress": int(task.progress or 0),
                "message": task.message or "Conversion in progress...",
            }
        elif task.status in ("error", "interrupted"):
            return {
                "step": "error",
                "progress": 0,
                "message": task.message or "Conversion failed",
                "error": task.error,
            }

    # No DB task — check filesystem for previously completed GGUF files
    gguf_dir = (PROJECTS_DIR / str(project_id) / "gguf").resolve()
    if not str(gguf_dir).startswith(str(PROJECTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project path")
    if gguf_dir.exists():
        gguf_files = list(gguf_dir.glob("*.gguf"))
        if gguf_files:
            return {
                "step": "completed",
                "progress": 100,
                "message": "GGUF file ready for download",
                "gguf_filename": gguf_files[0].name,
                "gguf_size_mb": round(gguf_files[0].stat().st_size / (1024 * 1024), 1),
            }

    return {"step": "not_started", "progress": 0, "message": "No GGUF conversion in progress"}


@router.get("/export/{project_id}/gguf/download")
def download_gguf(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Download the GGUF file for a project."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    gguf_dir = (PROJECTS_DIR / str(project_id) / "gguf").resolve()
    if not str(gguf_dir).startswith(str(PROJECTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project path")
    if not gguf_dir.exists():
        raise HTTPException(status_code=404, detail="No GGUF file found. Run GGUF export first.")

    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise HTTPException(status_code=404, detail="No GGUF file found. Run GGUF export first.")

    gguf_file = gguf_files[0]
    filename = f"meowllm-{_sanitize_filename(project.name)}.gguf"

    return FileResponse(
        path=str(gguf_file),
        filename=filename,
        media_type="application/octet-stream",
    )


# ── Path-based model routes (catch-all — MUST come AFTER /export/) ──


@router.get("/{model_id:path}/status", response_model=ModelStatusResponse)
def model_status(
    model_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    # Auth required to prevent model enumeration
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)

    result = {
        "model_id": model_id,
        "is_cached": _is_model_cached(model_id),
        "size_gb": model["size_gb"] if model else None,
    }

    # Include download progress from DB if actively downloading
    task = _get_task_status(db, "download", model_id)
    if task and task.status in ("running", "queued"):
        result["download"] = {
            "status": "downloading",
            "progress": task.progress or 0.0,
            "message": task.message or "",
            "error": task.error,
            "started_at": task.created_at.timestamp() if task.created_at else None,
        }
    elif task and task.status == "completed":
        result["download"] = {
            "status": "completed",
            "progress": 100.0,
            "message": task.message or "",
            "local_path": (task.metadata_ or {}).get("local_path"),
            "completed_at": task.updated_at.timestamp() if task.updated_at else None,
        }
    elif task and task.status in ("error", "interrupted"):
        result["download"] = {
            "status": task.status,
            "progress": 0.0,
            "message": task.message or "",
            "error": task.error,
        }

    return result


@router.post("/{model_id:path}/download", response_model=DownloadStartResponse)
def download_model(
    model_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Trigger real model download from HuggingFace Hub in a background thread."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    _cleanup_stale_tasks(db)

    if not _MODEL_ID_RE.match(model_id):
        raise HTTPException(status_code=400, detail="Invalid model ID format. Use 'org/model-name'.")

    model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)
    model_name = model["name"] if model else model_id.split("/")[-1]

    if _is_model_cached(model_id):
        return {"detail": "Model already cached", "status": "cached"}

    # Check if already downloading (in DB)
    existing = _get_task_status(db, "download", model_id)
    if existing and existing.status in ("running", "queued"):
        return {
            "detail": f"{model_name} is already downloading.",
            "status": "downloading",
            "progress": existing.progress or 0.0,
        }

    # Create DB task record BEFORE launching thread
    task = _get_or_create_task(
        db, "download", model_id,
        initial_status="running",
        initial_message=f"Starting download of {model_name}...",
        metadata={"model_name": model_name},
    )

    # Launch download in background thread
    thread = threading.Thread(
        target=_download_model_thread,
        args=(model_id, model_name),
        name=f"download-{model_id.replace('/', '-')}",
        daemon=True,
    )
    thread.start()

    return {
        "detail": f"Download started for {model_name}. Use /status endpoint to track progress.",
        "status": "downloading",
        "estimated_size_gb": model["size_gb"] if model else None,
    }


@router.get("/{model_id:path}/download/progress", response_model=DownloadProgressResponse)
def download_progress(
    model_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Get the progress of an ongoing model download."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)
    model_name = model["name"] if model else model_id.split("/")[-1]

    if _is_model_cached(model_id):
        return {
            "status": "cached",
            "progress": 100.0,
            "message": f"{model_name} is ready to use.",
            "is_cached": True,
        }

    # Check DB for task
    task = _get_task_status(db, "download", model_id)
    if task:
        result = {
            "status": task.status if task.status != "running" else "downloading",
            "progress": task.progress or 0.0,
            "message": task.message or "",
            "is_cached": False,
            "error": task.error,
            "started_at": task.created_at.timestamp() if task.created_at else None,
            "completed_at": task.updated_at.timestamp() if task.status in ("completed", "error") else None,
            "local_path": (task.metadata_ or {}).get("local_path"),
        }
        if task.created_at:
            elapsed = time.time() - task.created_at.timestamp()
            result["elapsed_seconds"] = round(elapsed)
        return result

    return {
        "status": "not_started",
        "progress": 0.0,
        "message": "No download in progress.",
        "is_cached": False,
    }


@router.delete("/{model_id:path}/download", response_model=DetailResponse)
def cancel_download(
    model_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Cancel an ongoing model download."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)
    model_name = model["name"] if model else model_id.split("/")[-1]

    task = (
        db.query(BackgroundTask)
        .filter(
            BackgroundTask.task_type == "download",
            BackgroundTask.task_key == model_id,
            BackgroundTask.status.in_(("running", "queued")),
        )
        .first()
    )
    if not task:
        raise HTTPException(status_code=400, detail="No active download to cancel")

    task.status = "cancelled"
    task.message = f"Download of {model_name} was cancelled."
    task.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {"detail": f"Download cancelled for {model_name}"}


@router.delete("/{model_id:path}/cache", response_model=DetailResponse)
def delete_cached_model(
    model_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Delete a cached model from disk to free space."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if not _MODEL_ID_RE.match(model_id):
        raise HTTPException(status_code=400, detail="Invalid model ID format. Use 'org/model-name'.")

    if not _is_model_cached(model_id):
        raise HTTPException(status_code=404, detail="Model is not cached locally")

    # Ensure no active download for this model
    active_task = (
        db.query(BackgroundTask)
        .filter(
            BackgroundTask.task_type == "download",
            BackgroundTask.task_key == model_id,
            BackgroundTask.status.in_(("running", "queued")),
        )
        .first()
    )
    if active_task:
        raise HTTPException(status_code=409, detail="Cannot delete while download is in progress")

    model_dir_name = "models--" + model_id.replace("/", "--")
    deleted = False
    freed_bytes = 0

    # Remove from HuggingFace cache
    hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    hf_model_dir = hf_cache / model_dir_name
    if hf_model_dir.is_dir():
        freed_bytes += sum(f.stat().st_size for f in hf_model_dir.rglob("*") if f.is_file())
        shutil.rmtree(hf_model_dir, ignore_errors=True)
        deleted = True

    # Remove from app model cache
    app_model_dir = MODEL_CACHE_DIR / model_dir_name
    if app_model_dir.is_dir():
        freed_bytes += sum(f.stat().st_size for f in app_model_dir.rglob("*") if f.is_file())
        shutil.rmtree(app_model_dir, ignore_errors=True)
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Model cache directory not found on disk")

    # Clean up any related background tasks
    db.query(BackgroundTask).filter(
        BackgroundTask.task_type == "download",
        BackgroundTask.task_key == model_id,
    ).delete(synchronize_session=False)
    db.commit()

    freed_gb = round(freed_bytes / (1024 ** 3), 2)
    model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)
    model_name = model["name"] if model else model_id.split("/")[-1]
    logger.info("Deleted cached model %s (%s GB freed)", model_id, freed_gb)

    return {"detail": f"Deleted cached model {model_name}. Freed {freed_gb} GB."}


@router.get("/export/{project_id}/gguf/list", response_model=list[dict])
def list_gguf_files(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """List all GGUF files for a project."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    gguf_dir = (PROJECTS_DIR / str(project_id) / "gguf").resolve()
    if not str(gguf_dir).startswith(str(PROJECTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project path")

    if not gguf_dir.exists():
        return []

    result = []
    for f in sorted(gguf_dir.glob("*.gguf")):
        stat = f.stat()
        result.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 1),
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        })
    return result


@router.delete("/export/{project_id}/gguf/{filename}", response_model=DetailResponse)
def delete_gguf_file(
    project_id: int,
    filename: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Delete a specific GGUF file for a project."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate filename is safe (prevent path traversal)
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    if safe_filename != filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Only .gguf files can be deleted")

    gguf_dir = (PROJECTS_DIR / str(project_id) / "gguf").resolve()
    if not str(gguf_dir).startswith(str(PROJECTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid project path")

    file_path = gguf_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="GGUF file not found")

    size_mb = round(file_path.stat().st_size / (1024 * 1024), 1)
    file_path.unlink()
    logger.info("Deleted GGUF file %s for project %d (%s MB)", filename, project_id, size_mb)

    return {"detail": f"Deleted {filename} ({size_mb} MB)"}

