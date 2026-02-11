"""
Hardware monitoring service for MeowLLM Studio.

Provides real-time system stats including:
- CPU: usage %, per-core count, name
- RAM: total, available, usage %
- GPU (NVIDIA via pynvml): usage %, temp, VRAM, power draw, fan speed, memory bandwidth
- Disk: total, free, model cache size

Optimized for 0.5s polling — uses non-blocking cpu_percent() and
keeps pynvml handle as a module-level singleton.
"""

import psutil
import shutil
import platform
import time
from pathlib import Path
from app.config import MODEL_CACHE_DIR

# ── NVIDIA GPU singleton handle ──────────────────────────────────────
_nvml_initialized = False
_gpu_handle = None


def _ensure_nvml():
    """Initialize NVML once and cache the GPU handle."""
    global _nvml_initialized, _gpu_handle
    if _nvml_initialized:
        return _gpu_handle is not None
    _nvml_initialized = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception:
        _gpu_handle = None
        return False


def _get_gpu_stats() -> dict:
    """Get detailed GPU stats. Returns empty defaults if no GPU."""
    defaults = {
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
    if not _ensure_nvml() or _gpu_handle is None:
        return defaults

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
        util = pynvml.nvmlDeviceGetUtilizationRates(h)

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
            "gpu_usage_percent": util.gpu,
            "gpu_memory_utilization_percent": util.memory,
            "gpu_temp_celsius": temp,
            "gpu_power_watts": power_w,
            "gpu_power_limit_watts": power_limit_w,
            "gpu_fan_percent": fan,
        }
    except Exception:
        return defaults


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
