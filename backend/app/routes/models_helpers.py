"""
Shared helpers for model routes — DB-backed task management, validation,
cache checking, compatibility, HF API retry logic, and rate limiting.

Extracted from the monolithic models.py to improve maintainability.
"""

import os
import re
import threading
import logging
import time
import shutil
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.background_task import BackgroundTask
from app.config import MODEL_CACHE_DIR, MODELS_DIR

logger = logging.getLogger("meowllm.models")

# ── Constants ───────────────────────────────────────────────────────
_HF_API_TIMEOUT = 15      # seconds for HuggingFace API calls
_HF_MAX_RETRIES = 3       # retries for transient HF API failures
_DISK_HEADROOM_GB = 2.0   # leave this much disk free after download
_DOWNLOAD_STALE_HOURS = 6 # mark "running" downloads as stale if this old
_DOWNLOAD_TTL = 3600      # 1 hour — auto-cleanup completed/errored downloads
_REQUIRED_SNAPSHOT_FILES = {"config.json"}

# ── Rate limiting for HF lookups (per-IP, in-memory) ────────────────
_hf_lookup_times: dict[str, list[float]] = {}
_HF_LOOKUP_RATE_LIMIT = 10
_HF_LOOKUP_WINDOW = 60
_hf_rate_lock = threading.Lock()
_hf_purge_counter = 0
_HF_PURGE_INTERVAL = 100

# ── Validation ──────────────────────────────────────────────────────
_MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in filenames."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)[:200]


def check_hf_rate_limit(client_id: str) -> bool:
    """Return True if the client is within rate limits, False if throttled."""
    global _hf_purge_counter
    now = time.time()
    with _hf_rate_lock:
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
        times = [t for t in times if now - t < _HF_LOOKUP_WINDOW]
        if len(times) >= _HF_LOOKUP_RATE_LIMIT:
            _hf_lookup_times[client_id] = times
            return False
        times.append(now)
        _hf_lookup_times[client_id] = times
        return True


# ═══════════════════════════════════════════════════════════════════
#  DB-backed task helpers (replace the old in-memory dicts)
# ═══════════════════════════════════════════════════════════════════

def get_or_create_task(
    db: Session,
    task_type: str,
    task_key: str,
    *,
    initial_status: str = "running",
    initial_message: str = "",
    metadata: dict | None = None,
) -> BackgroundTask:
    """Get an existing active task or create a new one.

    Uses DB-level state, safe across multiple workers/processes.
    """
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


def update_task(task_type: str, task_key: str, **fields) -> None:
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


def finish_task(task_type: str, task_key: str, *, status: str, message: str = "", error: str | None = None, metadata: dict | None = None) -> None:
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


def get_task_status(db: Session, task_type: str, task_key: str) -> BackgroundTask | None:
    """Get the latest task for a given type/key (active first, then most recent)."""
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
    return (
        db.query(BackgroundTask)
        .filter(
            BackgroundTask.task_type == task_type,
            BackgroundTask.task_key == task_key,
        )
        .order_by(BackgroundTask.updated_at.desc())
        .first()
    )


def cleanup_stale_tasks(db: Session) -> None:
    """Remove completed/errored tasks older than TTL."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_DOWNLOAD_TTL)
    db.query(BackgroundTask).filter(
        BackgroundTask.status.in_(("completed", "error", "cancelled", "interrupted")),
        BackgroundTask.updated_at < cutoff,
    ).delete(synchronize_session=False)
    db.commit()


# ═══════════════════════════════════════════════════════════════════
#  Hardware & cache helpers
# ═══════════════════════════════════════════════════════════════════

def check_compatibility(model: dict, hw: dict) -> str:
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


def is_model_cached(model_id: str, *, deep_check: bool = False, extra_paths: list[str] | None = None) -> bool:
    """Check if a model is already cached locally with integrity validation.

    Searches (in order):
    1. Default HuggingFace hub cache (~/.cache/huggingface/hub) — HF snapshot format
    2. App MODEL_CACHE_DIR — HF snapshot format
    3. MODELS_DIR/<model_safe_name>/ — flat local_dir format (preferred new location)
    4. Any extra_paths provided (flat local_dir format)
    """
    model_dir_name = "models--" + model_id.replace("/", "--")
    model_safe_name = model_id.replace("/", "--")

    # ── 1. HuggingFace default cache (HF snapshot format) ───────────
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    hf_model_dir = os.path.join(hf_cache, model_dir_name)
    if os.path.isdir(hf_model_dir):
        if validate_snapshot_dir(hf_model_dir, deep_check=deep_check):
            return True

    # ── 2. App MODEL_CACHE_DIR (HF snapshot format) ──────────────────
    app_model_dir = MODEL_CACHE_DIR / model_dir_name
    if app_model_dir.is_dir():
        if validate_snapshot_dir(str(app_model_dir), deep_check=deep_check):
            return True

    # ── 3. MODELS_DIR flat local_dir format ──────────────────────────
    flat_dir = MODELS_DIR / model_safe_name
    if flat_dir.is_dir():
        if validate_flat_model_dir(str(flat_dir), deep_check=deep_check):
            return True

    # Also scan MODELS_DIR top-level for any sub-directory matching model_id variations
    if MODELS_DIR.is_dir():
        for candidate in MODELS_DIR.iterdir():
            if not candidate.is_dir():
                continue
            # Match by exact name or slash-replaced variants
            cname = candidate.name.lower()
            mid_variants = {
                model_id.lower(),
                model_id.lower().replace("/", "--"),
                model_id.lower().replace("/", "_"),
                model_id.split("/")[-1].lower(),
            }
            if cname in mid_variants:
                if validate_flat_model_dir(str(candidate), deep_check=deep_check):
                    return True
                if validate_snapshot_dir(str(candidate), deep_check=deep_check):
                    return True

    # ── 4. Any caller-supplied extra paths ───────────────────────────
    if extra_paths:
        for p in extra_paths:
            if not p or not os.path.isdir(p):
                continue
            if validate_flat_model_dir(p, deep_check=deep_check):
                return True
            if validate_snapshot_dir(p, deep_check=deep_check):
                return True

    return False


def get_model_local_path(model_id: str) -> str | None:
    """Return the local filesystem path of a cached model, or None if not cached.

    Prefers the flat MODELS_DIR structure over the HF cache structure so that
    the returned path can be used directly with from_pretrained().
    """
    model_dir_name = "models--" + model_id.replace("/", "--")
    model_safe_name = model_id.replace("/", "--")

    # ── MODELS_DIR flat format (preferred) ───────────────────────────
    flat_dir = MODELS_DIR / model_safe_name
    if flat_dir.is_dir() and validate_flat_model_dir(str(flat_dir)):
        return str(flat_dir)

    # Scan MODELS_DIR for variants
    if MODELS_DIR.is_dir():
        for candidate in MODELS_DIR.iterdir():
            if not candidate.is_dir():
                continue
            cname = candidate.name.lower()
            mid_variants = {
                model_id.lower(),
                model_id.lower().replace("/", "--"),
                model_id.split("/")[-1].lower(),
            }
            if cname in mid_variants:
                if validate_flat_model_dir(str(candidate)):
                    return str(candidate)

    # ── HF snapshot caches — return the snapshot sub-directory ───────
    for base_dir in [os.path.expanduser("~/.cache/huggingface/hub"), str(MODEL_CACHE_DIR)]:
        hf_model_dir = os.path.join(base_dir, model_dir_name)
        if os.path.isdir(hf_model_dir) and validate_snapshot_dir(hf_model_dir):
            snapshots_dir = os.path.join(hf_model_dir, "snapshots")
            try:
                snaps = sorted(os.listdir(snapshots_dir))
                if snaps:
                    return os.path.join(snapshots_dir, snaps[-1])
            except OSError:
                pass

    return None


def validate_snapshot_dir(model_dir: str, *, deep_check: bool = False) -> bool:
    """Validate that a model cache directory has a complete HF snapshot."""
    try:
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            return False

        snapshots = [d for d in os.listdir(snapshots_dir)
                     if os.path.isdir(os.path.join(snapshots_dir, d))]
        if not snapshots:
            return False

        latest_snapshot = os.path.join(snapshots_dir, snapshots[-1])
        snapshot_files = set(os.listdir(latest_snapshot))

        if not snapshot_files:
            logger.warning("Empty snapshot directory: %s", latest_snapshot)
            return False

        missing = _REQUIRED_SNAPSHOT_FILES - snapshot_files
        if missing:
            logger.warning("Snapshot missing required files %s: %s", missing, latest_snapshot)
            return False

        if deep_check:
            for fname in snapshot_files:
                fpath = os.path.join(latest_snapshot, fname)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    if size == 0:
                        logger.warning("Empty file in snapshot: %s", fpath)
                        return False

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


def validate_flat_model_dir(model_dir: str, *, deep_check: bool = False) -> bool:
    """Validate a model directory downloaded with local_dir (flat structure).

    In this format files sit directly inside model_dir/ (no snapshots/ sub-tree).
    Minimum requirement: a non-empty, valid config.json must be present.
    """
    try:
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            return False
        if os.path.getsize(config_path) == 0:
            logger.warning("Empty config.json in flat model dir: %s", model_dir)
            return False

        import json as _json
        try:
            with open(config_path, "r") as f:
                _json.load(f)
        except (_json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt config.json in flat model dir %s: %s", model_dir, e)
            return False

        if deep_check:
            # Extra validation: every file should be non-empty
            for entry in os.scandir(model_dir):
                if entry.is_file() and entry.stat().st_size == 0:
                    logger.warning("Empty file in flat model dir: %s", entry.path)
                    return False

        return True
    except OSError as e:
        logger.debug("Error validating flat model dir %s: %s", model_dir, e)
        return False


def get_disk_free_gb(path: str | None = None) -> float:
    """Get free disk space in GB at the given path (defaults to MODELS_DIR).

    Falls back gracefully through: given path → MODELS_DIR → HF cache dir → /
    """
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(str(MODELS_DIR))
    candidates.append(os.path.expanduser("~/.cache/huggingface/hub"))
    candidates.append("/")

    for p in candidates:
        try:
            os.makedirs(p, exist_ok=True)
            usage = shutil.disk_usage(p)
            return round(usage.free / (1024 ** 3), 2)
        except OSError:
            continue
    return 0.0


def check_hf_reachable() -> tuple[bool, str]:
    """Quick connectivity check to HuggingFace Hub."""
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


def hf_api_call_with_retry(fn, *args, max_retries: int = _HF_MAX_RETRIES, **kwargs):
    """Call a HuggingFace Hub API function with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e)
            if any(code in error_str for code in ("401", "403", "404", "422")):
                raise
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + 0.5
                logger.info("HF API attempt %d/%d failed (%s), retrying in %.1fs...",
                           attempt + 1, max_retries, error_str[:80], wait)
                time.sleep(wait)
    raise last_error
