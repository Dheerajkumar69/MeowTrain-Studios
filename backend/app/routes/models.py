import os
import re
import threading
import logging
import time
import shutil
import zipfile
import hashlib
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.request import urlopen, Request
from urllib.error import URLError

from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks, Query
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

# Import shared helpers (extracted to reduce file size)
from app.routes.models_helpers import (
    sanitize_filename,
    check_hf_rate_limit,
    get_or_create_task,
    update_task,
    finish_task,
    get_task_status,
    cleanup_stale_tasks,
    check_compatibility,
    is_model_cached,
    validate_snapshot_dir,
    get_disk_free_gb,
    check_hf_reachable,
    hf_api_call_with_retry,
    _MODEL_ID_RE,
    _HF_API_TIMEOUT,
    _HF_MAX_RETRIES,
    _DISK_HEADROOM_GB,
    _DOWNLOAD_STALE_HOURS,
    _DOWNLOAD_TTL,
    _REQUIRED_SNAPSHOT_FILES,
)

logger = logging.getLogger("meowllm.models")

router = APIRouter(prefix="/models", tags=["Models"])

# Backward-compatible aliases for internal usage throughout this file
_sanitize_filename = sanitize_filename
_check_hf_rate_limit = check_hf_rate_limit
_get_or_create_task = get_or_create_task
_update_task = update_task
_finish_task = finish_task
_get_task_status = get_task_status
_cleanup_stale_tasks = cleanup_stale_tasks
_check_compatibility = check_compatibility
_is_model_cached = is_model_cached
_validate_snapshot_dir = validate_snapshot_dir
_get_disk_free_gb = get_disk_free_gb
_check_hf_reachable = check_hf_reachable
_hf_api_call_with_retry = hf_api_call_with_retry

# Note: In-memory locks removed — DB-backed BackgroundTask model provides
# cross-process safe task management (works with gunicorn -w N).
_DOWNLOAD_TTL_LOCAL = _DOWNLOAD_TTL  # Alias for local reference

# ── Constants ───────────────────────────────────────────────────────
_HF_API_TIMEOUT = 15      # seconds for HuggingFace API calls
_HF_MAX_RETRIES = 3       # retries for transient HF API failures
_DISK_HEADROOM_GB = 2.0   # leave this much disk free after download
_DOWNLOAD_STALE_HOURS = 6 # mark "running" downloads as stale if this old
_REQUIRED_SNAPSHOT_FILES = {"config.json"}  # files that MUST exist in a valid cache

# ── Rate limiting for HF lookups (per-IP, in-memory) ────────────────
_hf_lookup_times: dict[str, list[float]] = {}  # ip → [timestamps]
_HF_LOOKUP_RATE_LIMIT = 10     # lookups per window
_HF_LOOKUP_WINDOW = 60         # seconds
_hf_rate_lock = threading.Lock()
_hf_purge_counter = 0          # purge stale IPs every N calls
_HF_PURGE_INTERVAL = 100       # purge after this many rate checks


def _check_hf_rate_limit(client_id: str) -> bool:
    """Return True if the client is within rate limits, False if throttled."""
    global _hf_purge_counter
    now = time.time()
    with _hf_rate_lock:
        # Periodic purge of stale IPs to prevent memory leak
        _hf_purge_counter += 1
        if _hf_purge_counter >= _HF_PURGE_INTERVAL:
            _hf_purge_counter = 0
            stale_ips = [
                ip for ip, times in _hf_lookup_times.items()
                if not times or (now - max(times)) > _HF_LOOKUP_WINDOW * 2
            ]
            for ip in stale_ips:
                del _hf_lookup_times[ip]

        times = _hf_lookup_times.get(client_id, [])
        # Prune old entries
        times = [t for t in times if now - t < _HF_LOOKUP_WINDOW]
        if len(times) >= _HF_LOOKUP_RATE_LIMIT:
            _hf_lookup_times[client_id] = times
            return False
        times.append(now)
        _hf_lookup_times[client_id] = times
        return True


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


def _is_model_cached(model_id: str, *, deep_check: bool = False) -> bool:
    """Check if a model is already cached locally with integrity validation.
    
    A cached model is valid only if:
    1. The snapshot directory exists with at least one snapshot
    2. The snapshot contains required files (config.json at minimum)
    3. optionally (deep_check): files are non-empty and not truncated
    """
    model_dir_name = "models--" + model_id.replace("/", "--")
    
    # Check HuggingFace cache first
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    hf_model_dir = os.path.join(hf_cache, model_dir_name)
    if os.path.isdir(hf_model_dir):
        if _validate_snapshot_dir(hf_model_dir, deep_check=deep_check):
            return True
    
    # Check app-local model cache
    app_model_dir = MODEL_CACHE_DIR / model_dir_name
    if app_model_dir.is_dir():
        if _validate_snapshot_dir(str(app_model_dir), deep_check=deep_check):
            return True
    
    return False


def _validate_snapshot_dir(model_dir: str, *, deep_check: bool = False) -> bool:
    """Validate that a model cache directory has a complete snapshot."""
    try:
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return False
        
        snapshots = [d for d in os.listdir(snapshots_dir) 
                     if os.path.isdir(os.path.join(snapshots_dir, d))]
        if not snapshots:
            return False
        
        # Use the most recent snapshot (sorted by name = hash)
        latest_snapshot = os.path.join(snapshots_dir, snapshots[-1])
        snapshot_files = set(os.listdir(latest_snapshot))
        
        if not snapshot_files:
            logger.warning("Empty snapshot directory: %s", latest_snapshot)
            return False
        
        # Check required files exist
        missing = _REQUIRED_SNAPSHOT_FILES - snapshot_files
        if missing:
            logger.warning("Snapshot missing required files %s: %s", missing, latest_snapshot)
            return False
        
        if deep_check:
            # Verify files are non-empty (catches truncated downloads)
            for fname in snapshot_files:
                fpath = os.path.join(latest_snapshot, fname)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    if size == 0:
                        logger.warning("Empty file in snapshot: %s", fpath)
                        return False
            
            # Verify config.json is valid JSON
            config_path = os.path.join(latest_snapshot, "config.json")
            if os.path.isfile(config_path):
                try:
                    import json
                    with open(config_path, "r") as f:
                        json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Corrupt config.json in snapshot: %s (%s)", config_path, e)
                    return False
        
        return True
    except OSError as e:
        logger.debug("Error validating snapshot %s: %s", model_dir, e)
        return False


def _get_disk_free_gb() -> float:
    """Get free disk space in GB where models are cached."""
    try:
        # Check the HuggingFace cache partition
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(hf_cache, exist_ok=True)
        usage = shutil.disk_usage(hf_cache)
        return round(usage.free / (1024 ** 3), 2)
    except OSError:
        try:
            usage = shutil.disk_usage("/")
            return round(usage.free / (1024 ** 3), 2)
        except OSError:
            return 0.0


def _check_hf_reachable() -> tuple[bool, str]:
    """Quick connectivity check to HuggingFace Hub. Returns (reachable, message)."""
    try:
        req = Request("https://huggingface.co/api/models?limit=1", method="HEAD")
        req.add_header("User-Agent", "MeowTrain/1.0")
        with urlopen(req, timeout=10) as resp:
            if resp.status < 400:
                return True, "HuggingFace Hub reachable"
            return False, f"HuggingFace returned HTTP {resp.status}"
    except URLError as e:
        reason = str(getattr(e, "reason", e))
        if "SSL" in reason or "certificate" in reason.lower():
            return False, "SSL/TLS error connecting to HuggingFace. Check system certificates."
        return False, f"Cannot reach HuggingFace Hub: {reason}"
    except Exception as e:
        return False, f"Cannot reach HuggingFace Hub: {e}"


def _hf_api_call_with_retry(fn, *args, max_retries: int = _HF_MAX_RETRIES, **kwargs):
    """Call a HuggingFace Hub API function with retry logic for transient failures."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Don't retry on permanent errors
            if any(code in error_str for code in ("401", "403", "404", "422")):
                raise
            # Retry on transient errors (429, 500, 502, 503, timeout)
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + 0.5  # 1.5s, 2.5s, 4.5s
                logger.info("HF API attempt %d/%d failed (%s), retrying in %.1fs...",
                           attempt + 1, max_retries, error_str[:80], wait)
                time.sleep(wait)
    raise last_error


def _download_model_thread(model_id: str, model_name: str, estimated_size_gb: float = 0.0):
    """Background thread that downloads a model from HuggingFace Hub.
    
    Hardened with:
    - Disk space pre-check (with headroom)
    - HuggingFace connectivity check before starting
    - Real-time progress with download speed
    - Cancellation watchdog via DB polling
    - Post-download integrity verification
    - Partial download cleanup on failure
    - Detailed error categorization & user-friendly messages
    """
    download_start_time = time.time()
    local_path = None
    
    try:
        with _download_lock:
            _update_task("download", model_id,
                         status="running",
                         progress=0.0,
                         message=f"Starting download of {model_name}...")

        # ── Step 1: Connectivity check ───────────────────────────────
        _update_task("download", model_id,
                     progress=1.0,
                     message=f"Checking HuggingFace connectivity...")
        
        reachable, reason = _check_hf_reachable()
        if not reachable:
            _finish_task("download", model_id,
                         status="error",
                         message=f"Cannot reach HuggingFace Hub: {reason}",
                         error=reason)
            return

        # ── Step 2: Disk space check ─────────────────────────────────
        if estimated_size_gb > 0:
            free_gb = _get_disk_free_gb()
            required_gb = estimated_size_gb + _DISK_HEADROOM_GB
            if free_gb < required_gb:
                error_msg = (
                    f"Not enough disk space for {model_name}. "
                    f"Need ~{required_gb:.1f} GB but only {free_gb:.1f} GB free. "
                    f"Free up {required_gb - free_gb:.1f} GB or delete unused model caches."
                )
                _finish_task("download", model_id,
                             status="error",
                             message=error_msg,
                             error=error_msg)
                return

        # ── Step 3: Import dependencies ──────────────────────────────
        _update_task("download", model_id,
                     progress=3.0,
                     message=f"Preparing download of {model_name}...")
        
        try:
            from app.utils.lazy_imports import huggingface_hub as _hf_hub
            snapshot_download = _hf_hub().snapshot_download
            from tqdm.auto import tqdm as _tqdm_base
        except ImportError as e:
            error_msg = f"Missing dependency: {e}. Run: pip install huggingface-hub"
            _finish_task("download", model_id, status="error",
                         message=error_msg, error=error_msg)
            return

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        _update_task("download", model_id,
                     progress=5.0,
                     message=f"Downloading {model_name} from HuggingFace...")

        logger.info("Starting download: %s (est. %.1f GB, %.1f GB free)",
                    model_id, estimated_size_gb, _get_disk_free_gb())

        # ── Real-time progress tracking via custom tqdm class ────────
        _progress_state = {
            "downloaded": 0,
            "total": 0,
            "last_db_update": 0.0,
            "file_totals": {},   # tqdm-id → total bytes for that bar
            "file_done": {},     # tqdm-id → bytes downloaded so far
            "files_completed": 0,
            "files_total": 0,
        }
        _DB_UPDATE_INTERVAL = 2.0

        class _DownloadProgressTracker(_tqdm_base):
            """tqdm wrapper that reports aggregate download progress to the DB."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                bar_id = id(self)
                if self.total and self.total > 0:
                    _progress_state["file_totals"][bar_id] = self.total
                    _progress_state["file_done"][bar_id] = 0
                    _progress_state["total"] = sum(_progress_state["file_totals"].values())
                    _progress_state["files_total"] += 1

            def update(self, n=1):
                super().update(n)
                bar_id = id(self)
                if bar_id in _progress_state["file_done"]:
                    _progress_state["file_done"][bar_id] += n
                    _progress_state["downloaded"] = sum(_progress_state["file_done"].values())

                    now = time.time()
                    if now - _progress_state["last_db_update"] >= _DB_UPDATE_INTERVAL:
                        _progress_state["last_db_update"] = now
                        total = _progress_state["total"]
                        done = _progress_state["downloaded"]
                        if total > 0:
                            pct = 5.0 + (done / total) * 85.0  # 5–90%
                            done_mb = done / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            elapsed = now - download_start_time
                            speed_mbps = (done_mb / elapsed) if elapsed > 0 else 0
                            eta_str = ""
                            if speed_mbps > 0:
                                remaining_mb = total_mb - done_mb
                                eta_secs = remaining_mb / speed_mbps
                                if eta_secs < 60:
                                    eta_str = f" • ETA {eta_secs:.0f}s"
                                else:
                                    eta_str = f" • ETA {eta_secs / 60:.0f}m"
                            _update_task(
                                "download", model_id,
                                progress=round(min(pct, 90.0), 1),
                                message=(
                                    f"Downloading {model_name}: {done_mb:.0f}/{total_mb:.0f} MB "
                                    f"({pct:.0f}%) • {speed_mbps:.1f} MB/s{eta_str}"
                                ),
                            )

            def close(self):
                bar_id = id(self)
                if bar_id in _progress_state["file_done"]:
                    _progress_state["files_completed"] += 1
                super().close()
                _progress_state["file_totals"].pop(bar_id, None)
                _progress_state["file_done"].pop(bar_id, None)

        # ── Cancellation watchdog ────────────────────────────────────
        cancel_event = threading.Event()

        def _check_cancelled():
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

        def _cancellation_watchdog():
            while not cancel_event.is_set():
                if _check_cancelled():
                    cancel_event.set()
                    logger.info("Download cancellation detected for %s", model_id)
                    break
                # Also check disk space during download
                free_gb = _get_disk_free_gb()
                if free_gb < 0.5:
                    logger.error("Disk space critically low (%.2f GB) during download of %s",
                                free_gb, model_id)
                    _finish_task("download", model_id,
                                 status="error",
                                 message=f"Download stopped: disk space critically low ({free_gb:.1f} GB free)",
                                 error="Disk full")
                    cancel_event.set()
                    break
                cancel_event.wait(3)

        watchdog = threading.Thread(target=_cancellation_watchdog, daemon=True,
                                    name=f"cancel-watch-{model_id.replace('/', '-')}")
        watchdog.start()

        # ── Step 4: Download ──────────────────────────────────────────
        local_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            ignore_patterns=[
                "*.gguf", "*.bin", "*.msgpack",
                "consolidated.*", "original/**",
            ],
            tqdm_class=_DownloadProgressTracker,
        )

        # Final cancel check
        if cancel_event.is_set():
            _finish_task("download", model_id,
                         status="cancelled",
                         message=f"Download of {model_name} was cancelled.")
            logger.info("Download cancelled (post-completion): %s", model_id)
            return

        cancel_event.set()  # stop watchdog

        # ── Step 5: Integrity verification ────────────────────────────
        _update_task("download", model_id,
                     progress=92.0,
                     message=f"Verifying download integrity for {model_name}...")
        
        if not _is_model_cached(model_id, deep_check=True):
            error_msg = (
                f"Download of {model_name} completed but integrity check failed. "
                f"The download may be corrupted. Please delete the cache and retry."
            )
            logger.error("Integrity check failed for %s at %s", model_id, local_path)
            _finish_task("download", model_id,
                         status="error",
                         message=error_msg,
                         error=error_msg)
            return

        # ── Step 5b: Supply-chain safety — check for auto_map (remote code) ──
        supply_chain_warning = ""
        if local_path:
            import json as _json
            config_path = os.path.join(local_path, "config.json")
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as _f:
                        model_config = _json.load(_f)
                    if model_config.get("auto_map"):
                        from app.config import TRUST_REMOTE_CODE
                        auto_map_keys = list(model_config["auto_map"].keys())
                        if not TRUST_REMOTE_CODE:
                            supply_chain_warning = (
                                f" ⚠️ This model defines custom code via auto_map "
                                f"({', '.join(auto_map_keys)}). TRUST_REMOTE_CODE is disabled, "
                                f"so custom code will NOT be executed. If the model doesn't work "
                                f"correctly, you may need to enable it — but only if you trust "
                                f"the model author."
                            )
                            logger.warning(
                                "Model %s has auto_map entries: %s (TRUST_REMOTE_CODE=false)",
                                model_id, auto_map_keys,
                            )
                except Exception as e:
                    logger.debug("Could not read config.json for %s: %s", model_id, e)

        elapsed = time.time() - download_start_time
        elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.0f}s"

        _finish_task("download", model_id,
                     status="completed",
                     message=f"{model_name} downloaded successfully! ({elapsed_str}){supply_chain_warning}",
                     metadata={"local_path": local_path, "elapsed_seconds": round(elapsed)})

        logger.info("Download completed: %s -> %s (%.0fs)", model_id, local_path, elapsed)

    except Exception as e:
        error_msg = str(e)
        error_category = "unknown"

        if "401" in error_msg or "Unauthorized" in error_msg:
            error_category = "auth"
            error_msg = (
                f"Access denied for {model_name}. This is a gated model that requires "
                f"a HuggingFace access token. Set the HF_TOKEN environment variable "
                f"and ensure you've accepted the model's license on huggingface.co."
            )
        elif "403" in error_msg or "Forbidden" in error_msg:
            error_category = "auth"
            error_msg = (
                f"Forbidden: you don't have access to {model_name}. "
                f"Visit https://huggingface.co/{model_id} to request access, "
                f"then set HF_TOKEN in your environment."
            )
        elif "404" in error_msg:
            error_category = "not_found"
            error_msg = f"Model {model_id} not found on HuggingFace Hub. Check the model ID."
        elif "429" in error_msg or "rate" in error_msg.lower():
            error_category = "rate_limit"
            error_msg = (
                f"HuggingFace rate limit exceeded while downloading {model_name}. "
                f"Wait a few minutes and try again. Set HF_TOKEN for higher limits."
            )
        elif "disk" in error_msg.lower() or "space" in error_msg.lower() or "No space" in error_msg:
            error_category = "disk"
            free_gb = _get_disk_free_gb()
            error_msg = f"Not enough disk space to download {model_name}. {free_gb:.1f} GB free."
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            error_category = "timeout"
            error_msg = (
                f"Download of {model_name} timed out. Check your internet connection and try again."
            )
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            error_category = "network"
            error_msg = (
                f"Network error while downloading {model_name}. "
                f"Check your internet connection and try again."
            )
        elif "SSL" in error_msg or "certificate" in error_msg.lower():
            error_category = "ssl"
            error_msg = (
                f"SSL/TLS error downloading {model_name}. "
                f"Check system certificates or try: pip install --upgrade certifi"
            )

        logger.error("Download failed for %s [%s]: %s", model_id, error_category, error_msg)

        _finish_task("download", model_id,
                     status="error",
                     message=error_msg,
                     error=error_msg,
                     metadata={"error_category": error_category})


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


@router.get("/search")
def search_hf_models(
    q: str = Query("", min_length=2, max_length=200, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    sort: str = Query("downloads", description="Sort by: downloads, likes, trending"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Search HuggingFace Hub for models. Returns lightweight summaries for browsing.
    
    Hardened with:
    - Per-user rate limiting
    - Retry logic for HF API
    - Timeout protection
    - Result sanitization
    """
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    client_id = f"user-{user.id}"
    if not _check_hf_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many searches. Please wait a minute before trying again."
        )

    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Search query is required.")

    try:
        from app.utils.lazy_imports import huggingface_hub as _hf_hub
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        valid_sorts = {"downloads": "downloads", "likes": "likes", "trending": "trending"}
        sort_key = valid_sorts.get(sort, "downloads")

        results = _hf_api_call_with_retry(
            _hf_hub().list_models,
            search=q,
            limit=limit,
            sort=sort_key,
            direction=-1,
            token=hf_token,
            # Only text-generation-like models
            filter="text-generation",
        )

        models = []
        for m in results:
            model_id = m.id or ""
            if not model_id or not _MODEL_ID_RE.match(model_id):
                continue
            
            # Estimate size
            size_bytes = 0
            if hasattr(m, 'siblings') and m.siblings:
                size_bytes = sum(s.size for s in m.siblings if s.size)
            size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0.0
            
            # Parameter count
            params = "Unknown"
            if hasattr(m, 'safetensors') and m.safetensors:
                try:
                    total_params = sum(m.safetensors.get('parameters', {}).values())
                    if total_params >= 1e9:
                        params = f"{total_params / 1e9:.1f}B"
                    elif total_params >= 1e6:
                        params = f"{total_params / 1e6:.0f}M"
                except Exception:
                    pass
            if params == "Unknown":
                for tag in (getattr(m, 'tags', None) or []):
                    if tag.startswith("params:"):
                        params = tag.split(":")[1]
                        break
            if params == "Unknown":
                param_match = re.search(r'(\d+(?:\.\d+)?)[Bb]\b', model_id)
                if param_match:
                    params = f"{param_match.group(1)}B"

            models.append({
                "model_id": model_id,
                "name": model_id.split("/")[-1],
                "pipeline": getattr(m, 'pipeline_tag', None) or "unknown",
                "downloads": getattr(m, 'downloads', 0) or 0,
                "likes": getattr(m, 'likes', 0) or 0,
                "parameters": params,
                "size_gb": size_gb,
                "is_cached": _is_model_cached(model_id),
            })

        return {"query": q, "results": models, "total": len(models)}

    except ImportError:
        raise HTTPException(status_code=503, detail="huggingface-hub not installed")
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            raise HTTPException(status_code=429, detail="HuggingFace rate limit. Wait and retry.")
        logger.error("HF search failed for '%s': %s", q, error_msg)
        raise HTTPException(status_code=502, detail=f"Search failed: {error_msg[:200]}")


@router.get("/preflight")
def download_preflight(
    model_id: str = Query("", description="Model ID to check (optional)"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Pre-flight check for model downloading readiness.
    
    Returns:
    - HuggingFace connectivity status
    - Disk space status
    - HF token status
    - Model cache status (if model_id provided)
    """
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Connectivity
    hf_reachable, hf_message = _check_hf_reachable()

    # Disk space
    free_gb = _get_disk_free_gb()
    disk_ok = free_gb >= _DISK_HEADROOM_GB

    # HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    has_token = bool(hf_token)

    result = {
        "ready": hf_reachable and disk_ok,
        "hf_reachable": hf_reachable,
        "hf_message": hf_message,
        "disk_free_gb": free_gb,
        "disk_ok": disk_ok,
        "disk_headroom_gb": _DISK_HEADROOM_GB,
        "has_hf_token": has_token,
    }

    # Model-specific checks
    model_id = (model_id or "").strip()
    if model_id and _MODEL_ID_RE.match(model_id):
        model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)
        estimated_gb = model["size_gb"] if model else 0.0
        result["model_id"] = model_id
        result["model_cached"] = _is_model_cached(model_id)
        result["estimated_size_gb"] = estimated_gb
        if estimated_gb > 0:
            result["enough_space"] = free_gb >= estimated_gb + _DISK_HEADROOM_GB
        else:
            result["enough_space"] = disk_ok

    return result


@router.post("/custom/lookup", response_model=ModelInfo)
def lookup_custom_model(
    model_id: str = "",
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Look up a custom HuggingFace model by ID and return its info.
    
    Hardened with:
    - Per-user rate limiting (10 lookups/min)
    - Retry logic for transient HF API failures
    - Timeout protection
    - Model type validation (rejects non-text-generation models)
    - Better error messages for gated/private models
    """
    # Auth required to prevent anonymous HF API abuse
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Rate limit per user
    client_id = f"user-{user.id}"
    if not _check_hf_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many model lookups. Please wait a minute before trying again."
        )

    model_id = (model_id or "").strip()
    if not model_id or not _MODEL_ID_RE.match(model_id):
        raise HTTPException(
            status_code=400,
            detail="Model ID must be in 'org/name' format using only alphanumeric, dots, hyphens, and underscores "
                   "(e.g. 'meta-llama/Llama-3.2-3B')."
        )
    
    # Reject obviously wrong model IDs (too long, suspicious patterns)
    if len(model_id) > 200:
        raise HTTPException(status_code=400, detail="Model ID is too long.")

    try:
        from app.utils.lazy_imports import huggingface_hub as _hf_hub
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # Use retry wrapper for transient failures
        info = _hf_api_call_with_retry(
            _hf_hub().model_info,
            model_id,
            token=hf_token,
            timeout=_HF_API_TIMEOUT,
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="HuggingFace Hub library is not installed. Run: pip install huggingface-hub"
        )
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied for '{model_id}'. Set HF_TOKEN env variable "
                       f"and accept the license at https://huggingface.co/{model_id}"
            )
        if "403" in error_msg or "Forbidden" in error_msg:
            raise HTTPException(
                status_code=403,
                detail=f"'{model_id}' is a gated/private model. Visit "
                       f"https://huggingface.co/{model_id} to request access, "
                       f"then set HF_TOKEN in your .env file."
            )
        if "404" in error_msg:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found on HuggingFace Hub. "
                       f"Check the spelling and try again."
            )
        if "429" in error_msg or "rate" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="HuggingFace rate limit exceeded. Wait a minute and try again. "
                       "Set HF_TOKEN for higher limits."
            )
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail="HuggingFace API timed out. Please try again."
            )
        if "connection" in error_msg.lower() or "network" in error_msg.lower():
            raise HTTPException(
                status_code=502,
                detail="Cannot reach HuggingFace Hub. Check your internet connection."
            )
        logger.error("HF lookup failed for '%s': %s", model_id, error_msg)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to look up model '{model_id}': {error_msg[:200]}"
        )

    # Warn (but don't block) if model isn't a text generation model
    pipeline = info.pipeline_tag or "unknown"
    trainable_pipelines = {
        "text-generation", "text2text-generation", "conversational",
        "fill-mask", "feature-extraction", None, "unknown",
    }
    pipeline_warning = ""
    if pipeline not in trainable_pipelines:
        pipeline_warning = f" ⚠️ This is a '{pipeline}' model — fine-tuning may not work as expected."

    # Estimate size from model info — filter out non-model files
    model_extensions = {".safetensors", ".bin", ".pt", ".h5", ".onnx", ".msgpack"}
    size_bytes = 0
    file_count = 0
    for s in (info.siblings or []):
        if s.size:
            ext = os.path.splitext(s.rfilename or "")[1].lower()
            if ext in model_extensions or s.rfilename in ("config.json", "tokenizer.json", "tokenizer.model"):
                size_bytes += s.size
                file_count += 1
    # If no model files found, sum everything (fallback)
    if size_bytes == 0:
        size_bytes = sum(s.size for s in (info.siblings or []) if s.size)
    size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0.0

    # Try to determine parameter count from safetensors metadata, tags, or model card
    params = "Unknown"
    if hasattr(info, 'safetensors') and info.safetensors:
        try:
            total_params = sum(info.safetensors.get('parameters', {}).values())
            if total_params > 0:
                if total_params >= 1e9:
                    params = f"{total_params / 1e9:.1f}B"
                elif total_params >= 1e6:
                    params = f"{total_params / 1e6:.0f}M"
                else:
                    params = f"{total_params:,}"
        except Exception:
            pass
    if params == "Unknown":
        for tag in (info.tags or []):
            if tag.startswith("params:"):
                params = tag.split(":")[1]
                break
    # Also try to infer from model name (e.g. "Llama-3.2-3B" → "3B")
    if params == "Unknown":
        param_match = re.search(r'(\d+(?:\.\d+)?)[Bb]\b', model_id)
        if param_match:
            params = f"{param_match.group(1)}B"

    # Estimate resource requirements based on size (more accurate formula)
    if size_gb > 0:
        # Rule: ~2x model size for training RAM, ~1.2x for VRAM (with quantisation)
        ram_required = max(4, int(size_gb * 2.5))
        vram_required = max(2, int(size_gb * 1.2))
    else:
        ram_required = 8
        vram_required = 4

    hw = get_hardware_status()
    description = f"Custom model from HuggingFace Hub. Pipeline: {pipeline}.{pipeline_warning}"
    if info.downloads is not None:
        downloads_str = f"{info.downloads:,}" if info.downloads < 1_000_000 else f"{info.downloads / 1_000_000:.1f}M"
        description += f" Downloads: {downloads_str}."
    if info.likes is not None and info.likes > 0:
        description += f" ❤️ {info.likes:,} likes."

    model_dict = {
        "model_id": model_id,
        "name": model_id.split("/")[-1],
        "description": description,
        "parameters": params,
        "size_gb": size_gb,
        "ram_required_gb": ram_required,
        "vram_required_gb": vram_required,
        "recommended_hardware": f"{vram_required}GB+ VRAM or {ram_required}GB+ RAM",
        "estimated_train_minutes": max(10, int(size_gb * 5)) if size_gb > 0 else 30,
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
    """Trigger real model download from HuggingFace Hub in a background thread.
    
    Hardened with:
    - Disk space pre-check before starting download
    - Stale task cleanup
    - Deep cache check (verifies file integrity, not just directory existence)
    - Estimated size forwarded to download thread for ongoing disk checks
    """
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    _cleanup_stale_tasks(db)

    if not _MODEL_ID_RE.match(model_id):
        raise HTTPException(status_code=400, detail="Invalid model ID format. Use 'org/model-name'.")

    model = next((m for m in MODEL_CATALOG if m["model_id"] == model_id), None)
    model_name = model["name"] if model else model_id.split("/")[-1]
    estimated_size_gb = model["size_gb"] if model else 0.0

    if _is_model_cached(model_id):
        return {"detail": "Model already cached", "status": "cached"}

    # Pre-flight disk space check
    if estimated_size_gb > 0:
        free_gb = _get_disk_free_gb()
        required_gb = estimated_size_gb + _DISK_HEADROOM_GB
        if free_gb < required_gb:
            raise HTTPException(
                status_code=507,  # Insufficient Storage
                detail=f"Not enough disk space. Need ~{required_gb:.1f} GB but only "
                       f"{free_gb:.1f} GB free. Free up {required_gb - free_gb:.1f} GB "
                       f"or delete unused model caches."
            )

    # Check if already downloading (in DB)
    existing = _get_task_status(db, "download", model_id)
    if existing and existing.status in ("running", "queued"):
        # Check if the download is stale (thread crashed)
        if existing.updated_at:
            age_hours = (datetime.now(timezone.utc) - existing.updated_at).total_seconds() / 3600
            if age_hours > _DOWNLOAD_STALE_HOURS:
                logger.warning("Stale download detected for %s (%.1fh old), resetting",
                              model_id, age_hours)
                existing.status = "interrupted"
                existing.message = "Download appears stale (server may have restarted). Please retry."
                existing.error = "Stale download"
                existing.updated_at = datetime.now(timezone.utc)
                db.commit()
                # Fall through to start a new download
            else:
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
        metadata={"model_name": model_name, "estimated_size_gb": estimated_size_gb},
    )

    # Launch download in background thread
    thread = threading.Thread(
        target=_download_model_thread,
        args=(model_id, model_name, estimated_size_gb),
        name=f"download-{model_id.replace('/', '-')}",
        daemon=True,
    )
    thread.start()

    return {
        "detail": f"Download started for {model_name}. Use /status endpoint to track progress.",
        "status": "downloading",
        "estimated_size_gb": estimated_size_gb or None,
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


@router.post("/{model_id:path}/verify-cache")
def verify_model_cache(
    model_id: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Deep-verify a cached model's integrity. Returns detailed status."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if not _MODEL_ID_RE.match(model_id):
        raise HTTPException(status_code=400, detail="Invalid model ID format.")

    if not _is_model_cached(model_id, deep_check=False):
        return {"model_id": model_id, "cached": False, "valid": False,
                "message": "Model is not cached."}

    valid = _is_model_cached(model_id, deep_check=True)
    return {
        "model_id": model_id,
        "cached": True,
        "valid": valid,
        "message": "Cache is valid" if valid else "Cache appears corrupted. Delete and re-download.",
    }


def recover_interrupted_downloads():
    """Mark any 'running' download/gguf tasks as 'interrupted' on server startup.
    
    Called from main.py on_startup to handle downloads that were killed
    by a server restart.
    """
    db = SessionLocal()
    try:
        stale_tasks = (
            db.query(BackgroundTask)
            .filter(
                BackgroundTask.status.in_(("running", "queued")),
                BackgroundTask.task_type.in_(("download", "gguf")),
            )
            .all()
        )
        for task in stale_tasks:
            age = ""
            if task.updated_at:
                age_mins = (datetime.now(timezone.utc) - task.updated_at).total_seconds() / 60
                age = f" ({age_mins:.0f} min old)"
            logger.warning(
                "Recovering stale %s task '%s'%s — marking as interrupted",
                task.task_type, task.task_key, age
            )
            task.status = "interrupted"
            task.message = (
                f"This {task.task_type} was interrupted by a server restart. "
                f"Please retry."
            )
            task.error = "Server restarted during operation"
            task.updated_at = datetime.now(timezone.utc)
        if stale_tasks:
            db.commit()
            logger.info("Recovered %d interrupted tasks", len(stale_tasks))
    except Exception as e:
        logger.error("Failed to recover interrupted tasks: %s", e)
        db.rollback()
    finally:
        db.close()


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

