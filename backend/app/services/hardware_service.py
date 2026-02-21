"""
Hardware monitoring service for MeowLLM Studio.

Provides real-time system stats including:
- CPU: usage %, per-core count, name
- RAM: total, available, usage %
- GPU (NVIDIA via pynvml): usage %, temp, VRAM, power draw, fan speed, memory bandwidth
- Disk: total, free, model cache size

Optimized for 0.5s polling — uses non-blocking cpu_percent() and
keeps pynvml handle as a module-level singleton.

GPU detection strategy:
  1. Primary: pynvml (fast, reliable when installed)
  2. Fallback: nvidia-smi CLI parsing (always works if drivers are installed)
  3. Retries: if NVML init fails, retries every 30s (GPU may wake from suspend)
"""

import logging
import warnings

# Suppress pynvml deprecation warning (nvidia-ml-py re-exports the same API)
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)

import psutil
import shutil
import platform
import subprocess
import time
from pathlib import Path
from app.config import MODEL_CACHE_DIR

logger = logging.getLogger("meowllm.hardware")

# ── GPU defaults ─────────────────────────────────────────────────────
_GPU_DEFAULTS = {
    "gpu_available": False,
    "gpu_name": None,
    "gpu_vram_total_gb": None,
    "gpu_vram_available_gb": None,
    "gpu_vram_used_gb": None,
    "gpu_usage_percent": None,
    "gpu_memory_utilization_percent": None,
    "gpu_temp_celsius": None,
    "gpu_power_watts": None,
    "gpu_power_limit_watts": None,
    "gpu_fan_percent": None,
}

# ── NVIDIA GPU singleton handle ──────────────────────────────────────
_nvml_initialized = False
_gpu_handle = None
_nvml_last_attempt = 0.0
_NVML_RETRY_INTERVAL = 30.0  # seconds between retry attempts


def _ensure_nvml():
    """
    Initialize NVML and cache the GPU handle.

    Unlike the old version that gave up permanently on first failure,
    this retries every 30 seconds.  Covers:
      - pynvml not installed at startup but pip-installed later
      - GPU in suspend / power-save when server boots
      - Transient driver hiccups
    """
    global _nvml_initialized, _gpu_handle, _nvml_last_attempt

    # If already initialized successfully, validate the handle is still good
    if _nvml_initialized and _gpu_handle is not None:
        try:
            import pynvml
            # Quick health check — will throw if handle is stale
            pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
            return True
        except Exception:
            # Handle went stale (e.g. GPU reset, driver reload) — reinitialize
            logger.warning("NVML handle stale, reinitializing...")
            _nvml_initialized = False
            _gpu_handle = None

    # Throttle retry attempts so we don't spam logs / waste cycles
    now = time.time()
    if _nvml_initialized is False and _gpu_handle is None:
        if now - _nvml_last_attempt < _NVML_RETRY_INTERVAL:
            return False
    _nvml_last_attempt = now

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            logger.info("NVML initialized but no GPU devices found")
            _nvml_initialized = True
            _gpu_handle = None
            return False
        _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _nvml_initialized = True
        name = pynvml.nvmlDeviceGetName(_gpu_handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        logger.info("NVML initialized: %s", name)
        return True
    except ImportError:
        logger.debug("pynvml not installed — will use nvidia-smi fallback")
        _gpu_handle = None
        return False
    except Exception as e:
        logger.debug("NVML init failed (%s) — will retry in %.0fs", e, _NVML_RETRY_INTERVAL)
        _gpu_handle = None
        return False


def _get_gpu_stats_nvml() -> dict | None:
    """Get GPU stats via pynvml. Returns None if unavailable."""
    if not _ensure_nvml() or _gpu_handle is None:
        return None

    try:
        import pynvml
        h = _gpu_handle

        # Name
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Memory
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        vram_total = round(mem.total / (1024 ** 3), 2)
        vram_free = round(mem.free / (1024 ** 3), 2)
        vram_used = round((mem.total - mem.free) / (1024 ** 3), 2)

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util = util.gpu
            mem_util = util.memory
        except Exception:
            gpu_util = None
            mem_util = None

        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None

        # Power
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
            power_w = round(power_mw / 1000, 1)
        except Exception:
            power_w = None

        try:
            power_limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(h)
            power_limit_w = round(power_limit_mw / 1000, 1)
        except Exception:
            power_limit_w = None

        # Fan
        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(h)
        except Exception:
            fan = None

        return {
            "gpu_available": True,
            "gpu_name": name,
            "gpu_vram_total_gb": vram_total,
            "gpu_vram_available_gb": vram_free,
            "gpu_vram_used_gb": vram_used,
            "gpu_usage_percent": gpu_util,
            "gpu_memory_utilization_percent": mem_util,
            "gpu_temp_celsius": temp,
            "gpu_power_watts": power_w,
            "gpu_power_limit_watts": power_limit_w,
            "gpu_fan_percent": fan,
        }
    except Exception as e:
        # Handle went bad mid-query — force reinit next time
        logger.debug("NVML query failed: %s", e)
        _force_nvml_reinit()
        return None


def _force_nvml_reinit():
    """Reset NVML state so next call retries initialization."""
    global _nvml_initialized, _gpu_handle
    _nvml_initialized = False
    _gpu_handle = None


# ── nvidia-smi CLI fallback ──────────────────────────────────────────
_smi_cache: dict | None = None
_smi_cache_time = 0.0
_SMI_CACHE_TTL = 2.0  # seconds


def _get_gpu_stats_smi() -> dict | None:
    """
    Fallback GPU detection via nvidia-smi CLI.

    Slower than pynvml (~50ms subprocess) but always works if the
    NVIDIA driver is installed, even without pynvml.  Results are
    cached for 2 seconds to avoid subprocess spam.
    """
    global _smi_cache, _smi_cache_time
    now = time.time()
    if _smi_cache is not None and now - _smi_cache_time < _SMI_CACHE_TTL:
        return _smi_cache

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,"
                "utilization.gpu,utilization.memory,temperature.gpu,"
                "power.draw,power.limit,fan.speed",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            return None

        def _safe_float(val: str) -> float | None:
            try:
                v = val.strip()
                if v in ("[Not Supported]", "N/A", "[N/A]", ""):
                    return None
                return float(v)
            except (ValueError, TypeError):
                return None

        name = parts[0].strip()
        mem_total_mb = _safe_float(parts[1])
        mem_free_mb = _safe_float(parts[2])
        mem_used_mb = _safe_float(parts[3])
        gpu_util = _safe_float(parts[4])
        mem_util = _safe_float(parts[5])
        temp = _safe_float(parts[6]) if len(parts) > 6 else None
        power = _safe_float(parts[7]) if len(parts) > 7 else None
        power_limit = _safe_float(parts[8]) if len(parts) > 8 else None
        fan = _safe_float(parts[9]) if len(parts) > 9 else None

        stats = {
            "gpu_available": True,
            "gpu_name": name,
            "gpu_vram_total_gb": round(mem_total_mb / 1024, 2) if mem_total_mb else None,
            "gpu_vram_available_gb": round(mem_free_mb / 1024, 2) if mem_free_mb else None,
            "gpu_vram_used_gb": round(mem_used_mb / 1024, 2) if mem_used_mb else None,
            "gpu_usage_percent": int(gpu_util) if gpu_util is not None else None,
            "gpu_memory_utilization_percent": int(mem_util) if mem_util is not None else None,
            "gpu_temp_celsius": int(temp) if temp is not None else None,
            "gpu_power_watts": round(power, 1) if power is not None else None,
            "gpu_power_limit_watts": round(power_limit, 1) if power_limit is not None else None,
            "gpu_fan_percent": int(fan) if fan is not None else None,
        }

        _smi_cache = stats
        _smi_cache_time = now
        return stats

    except FileNotFoundError:
        # nvidia-smi not installed
        return None
    except Exception as e:
        logger.debug("nvidia-smi fallback failed: %s", e)
        return None


def _get_gpu_stats() -> dict:
    """
    Get GPU stats.  Tries pynvml first (fast), falls back to
    nvidia-smi CLI (always works with drivers), returns defaults
    if neither succeeds.
    """
    # Primary: pynvml (microseconds, in-process)
    stats = _get_gpu_stats_nvml()
    if stats is not None:
        return stats

    # Fallback: nvidia-smi CLI (milliseconds, subprocess)
    stats = _get_gpu_stats_smi()
    if stats is not None:
        return stats

    # No GPU detected
    return dict(_GPU_DEFAULTS)


# ── CPU name (cached, read once) ─────────────────────────────────────
_cpu_name_cache = None


def _get_cpu_name() -> str:
    global _cpu_name_cache
    if _cpu_name_cache:
        return _cpu_name_cache
    name = platform.processor() or "Unknown CPU"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    name = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    _cpu_name_cache = name
    return name


# ── Model cache size (cached) ────────────────────────────────────────
_model_cache_size_gb = 0.0
_model_cache_last_scan = 0.0
_MODEL_CACHE_SCAN_INTERVAL = 60.0  # seconds


def _get_cached_model_cache_size() -> float:
    """Get model cache size with time-based caching (refresh every 60s)."""
    global _model_cache_size_gb, _model_cache_last_scan
    now = time.time()
    if now - _model_cache_last_scan < _MODEL_CACHE_SCAN_INTERVAL:
        return _model_cache_size_gb
    try:
        if MODEL_CACHE_DIR.exists():
            total_size = sum(
                f.stat().st_size for f in MODEL_CACHE_DIR.rglob("*") if f.is_file()
            )
            _model_cache_size_gb = round(total_size / (1024 ** 3), 2)
        else:
            _model_cache_size_gb = 0.0
    except Exception:
        pass
    _model_cache_last_scan = now
    return _model_cache_size_gb


# ── Main API ─────────────────────────────────────────────────────────

def get_hardware_status() -> dict:
    """
    Return current hardware status.
    Uses non-blocking cpu_percent(interval=None) so this returns instantly
    (relies on psutil's internal delta tracking between calls).
    Safe for 0.5s polling loops.
    """
    # CPU — non-blocking (returns value since last call)
    cpu_name = _get_cpu_name()
    cpu_cores = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=None)

    # RAM
    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024 ** 3), 1)
    ram_available_gb = round(mem.available / (1024 ** 3), 1)
    ram_used_gb = round((mem.total - mem.available) / (1024 ** 3), 1)
    ram_usage_percent = mem.percent

    # GPU
    gpu = _get_gpu_stats()

    # Disk
    try:
        disk = shutil.disk_usage("/")
        disk_total_gb = round(disk.total / (1024 ** 3), 1)
        disk_free_gb = round(disk.free / (1024 ** 3), 1)
    except Exception:
        disk_total_gb = 0.0
        disk_free_gb = 0.0

    # Model cache size (cached for performance — only refresh every 60s)
    model_cache_size_gb = _get_cached_model_cache_size()

    return {
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "cpu_usage_percent": cpu_usage,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "ram_used_gb": ram_used_gb,
        "ram_usage_percent": ram_usage_percent,
        **gpu,
        "disk_total_gb": disk_total_gb,
        "disk_free_gb": disk_free_gb,
        "model_cache_size_gb": model_cache_size_gb,
    }
