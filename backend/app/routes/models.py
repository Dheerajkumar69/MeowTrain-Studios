import os
import threading
import logging
import time
import shutil
import zipfile
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.project import Project
from app.schemas import ModelInfo
from app.config import SUPPORTED_MODELS, MODEL_CACHE_DIR, PROJECTS_DIR
from app.services.hardware_service import get_hardware_status
from app.services.auth_service import get_user_from_header

logger = logging.getLogger("meowllm.models")

router = APIRouter(prefix="/models", tags=["Models"])

# ── Active download tracking ────────────────────────────────────────
_download_tasks: dict[str, dict] = {}
_download_lock = threading.Lock()
_DOWNLOAD_TTL = 3600  # 1 hour — auto-cleanup completed/errored downloads


def _cleanup_stale_downloads():
    """Remove completed/errored downloads older than TTL."""
    now = time.time()
    with _download_lock:
        stale = [
            mid for mid, info in _download_tasks.items()
            if info.get("status") in ("completed", "error")
            and now - info.get("completed_at", now) > _DOWNLOAD_TTL
        ]
        for mid in stale:
            del _download_tasks[mid]


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
        # Check if there are actual snapshot files (not just a partial download)
        snapshots_dir = os.path.join(hf_cache, model_dir_name, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                # Check the latest snapshot has files
                latest_snapshot = os.path.join(snapshots_dir, snapshots[-1])
                if os.listdir(latest_snapshot):
                    return True
    # Also check our custom cache
    if (MODEL_CACHE_DIR / model_dir_name).is_dir():
        return True
    return False


def _get_download_status(model_id: str) -> Optional[dict]:
    """Get the status of an ongoing download."""
    with _download_lock:
        return _download_tasks.get(model_id, None)


def _download_model_thread(model_id: str, model_name: str):
    """Background thread that downloads a model from HuggingFace Hub."""
    try:
        with _download_lock:
            _download_tasks[model_id] = {
                "status": "downloading",
                "progress": 0.0,
                "message": f"Starting download of {model_name}...",
                "started_at": time.time(),
                "error": None,
            }

        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError

        # Get HF token from environment for gated models
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        with _download_lock:
            _download_tasks[model_id]["message"] = f"Downloading {model_name} from HuggingFace..."
            _download_tasks[model_id]["progress"] = 5.0

        logger.info("Starting download: %s", model_id)

        # snapshot_download handles the download, caching, and resumption
        local_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            # Download only the files needed for inference/training
            ignore_patterns=[
                "*.gguf",         # Skip GGUF quantized versions
                "*.bin",          # Skip old-format bin files if safetensors exist
                "*.msgpack",      # Skip msgpack format
                "consolidated.*", # Skip consolidated format
                "original/**",    # Skip original format
            ],
        )

        with _download_lock:
            _download_tasks[model_id] = {
                "status": "completed",
                "progress": 100.0,
                "message": f"{model_name} downloaded successfully!",
                "local_path": local_path,
                "started_at": _download_tasks[model_id]["started_at"],
                "completed_at": time.time(),
                "error": None,
            }

        logger.info("Download completed: %s -> %s", model_id, local_path)

    except Exception as e:
        error_msg = str(e)

        # Check for specific errors
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

        with _download_lock:
            _download_tasks[model_id] = {
                "status": "error",
                "progress": 0.0,
                "message": error_msg,
                "error": error_msg,
                "started_at": _download_tasks.get(model_id, {}).get("started_at", time.time()),
                "completed_at": time.time(),
            }


@router.get("/", response_model=list[ModelInfo])
def list_models(db: Session = Depends(get_db)):
    hw = get_hardware_status()
    models = []
    for m in SUPPORTED_MODELS:
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


@router.get("/{model_id:path}/status")
def model_status(model_id: str):
    # Find model in registry
    model = next((m for m in SUPPORTED_MODELS if m["model_id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    result = {
        "model_id": model_id,
        "is_cached": _is_model_cached(model_id),
        "size_gb": model["size_gb"],
    }

    # Include download progress if actively downloading
    download_status = _get_download_status(model_id)
    if download_status:
        result["download"] = download_status

    return result


@router.post("/{model_id:path}/download")
def download_model(model_id: str):
    """Trigger real model download from HuggingFace Hub in a background thread."""
    _cleanup_stale_downloads()

    model = next((m for m in SUPPORTED_MODELS if m["model_id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    if _is_model_cached(model_id):
        return {"detail": "Model already cached", "status": "cached"}

    # Check if already downloading
    existing = _get_download_status(model_id)
    if existing and existing.get("status") == "downloading":
        return {
            "detail": f"{model['name']} is already downloading.",
            "status": "downloading",
            "progress": existing.get("progress", 0),
        }

    # Launch download in background thread
    thread = threading.Thread(
        target=_download_model_thread,
        args=(model_id, model["name"]),
        name=f"download-{model_id.replace('/', '-')}",
        daemon=True,
    )
    thread.start()

    return {
        "detail": f"Download started for {model['name']}. Use /status endpoint to track progress.",
        "status": "downloading",
        "estimated_size_gb": model["size_gb"],
    }


@router.get("/{model_id:path}/download/progress")
def download_progress(model_id: str):
    """Get the progress of an ongoing model download."""
    model = next((m for m in SUPPORTED_MODELS if m["model_id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    # Check if already cached
    if _is_model_cached(model_id):
        return {
            "status": "cached",
            "progress": 100.0,
            "message": f"{model['name']} is ready to use.",
            "is_cached": True,
        }

    # Check active download
    download_status = _get_download_status(model_id)
    if download_status:
        result = {**download_status, "is_cached": False}
        # Add elapsed time
        if "started_at" in download_status:
            elapsed = time.time() - download_status["started_at"]
            result["elapsed_seconds"] = round(elapsed)
        return result

    return {
        "status": "not_started",
        "progress": 0.0,
        "message": "No download in progress.",
        "is_cached": False,
    }


@router.delete("/{model_id:path}/download")
def cancel_download(model_id: str):
    """Cancel an ongoing model download."""
    model = next((m for m in SUPPORTED_MODELS if m["model_id"] == model_id), None)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in registry")

    with _download_lock:
        status = _download_tasks.get(model_id)
        if not status or status.get("status") != "downloading":
            raise HTTPException(status_code=400, detail="No active download to cancel")
        _download_tasks[model_id] = {
            "status": "cancelled",
            "progress": status.get("progress", 0),
            "message": f"Download of {model['name']} was cancelled.",
            "started_at": status.get("started_at", time.time()),
            "completed_at": time.time(),
            "error": None,
        }

    return {"detail": f"Download cancelled for {model['name']}"}


@router.get("/export/{project_id}")
def export_trained_model(
    project_id: int,
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

    # Check for adapters or checkpoints
    project_dir = PROJECTS_DIR / str(project_id)
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
    zip_filename = f"meowllm-{project.name.replace(' ', '_')}-model.zip"
    zip_path = project_dir / zip_filename

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(export_dir)
                    zf.write(file_path, arcname)

        return FileResponse(
            path=str(zip_path),
            filename=zip_filename,
            media_type="application/zip",
            background=None,  # Don't delete after send — user may re-download
        )
    except Exception as e:
        logger.error("Failed to create model export: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create model export")

