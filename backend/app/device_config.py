"""
Device auto-configuration for MeowTrain.

Reads the hardware config written by setup.py and provides
runtime helpers that adapt training behavior based on what's
available (CUDA, MPS, or CPU).

Also provides a startup check that warns if the wrong PyTorch
variant is installed (e.g., CPU torch on a machine with a GPU).
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("meowllm.device_config")

CONFIG_FILE = Path(__file__).resolve().parent.parent / ".device_config.json"

# Cached config — loaded once at startup, thread-safe
_device_config: Optional[dict] = None
_config_lock = threading.Lock()

# Config is considered stale after 30 days
_STALE_DAYS = 30

# Required keys for a valid config file
_REQUIRED_KEYS = {"mode", "training_device"}


def _detect_live() -> dict:
    """
    Live detection (fallback when .device_config.json doesn't exist).
    Runs quickly without subprocess calls — uses torch directly.
    Never raises — always returns a valid config dict.
    """
    config: dict = {
        "mode": "cpu",
        "training_device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "_source": "live_detection",
    }

    try:
        import torch
        config["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            config["mode"] = "cuda"
            config["training_device"] = "cuda"
            config["cuda_available"] = True
            try:
                config["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                config["vram_gb"] = round(props.total_memory / (1024**3), 1)
            except Exception as e:
                logger.warning("Could not query GPU properties: %s", e)
            config["cuda_version"] = torch.version.cuda
            config["gpu_count"] = torch.cuda.device_count()

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config["mode"] = "mps"
            config["training_device"] = "mps"
            config["mps_available"] = True

    except ImportError:
        logger.warning("PyTorch not installed — run `python setup.py` to configure")
    except Exception as e:
        logger.error("Unexpected error during live detection: %s", e)

    return config


def _validate_config(config: dict) -> bool:
    """Check that a config dict has all required keys and sensible values."""
    if not isinstance(config, dict):
        return False
    if not _REQUIRED_KEYS.issubset(config.keys()):
        return False
    if config.get("mode") not in ("cpu", "cuda", "mps", "rocm"):
        return False
    return True


def _is_stale(config: dict) -> bool:
    """Check if config is older than _STALE_DAYS."""
    detected_at = config.get("detected_at", "")
    if not detected_at:
        return False
    try:
        # Try multiple timestamp formats
        for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S"):
            try:
                import datetime
                dt = datetime.datetime.strptime(detected_at.strip(), fmt)
                age_days = (datetime.datetime.now() - dt).days
                return age_days > _STALE_DAYS
            except ValueError:
                continue
    except Exception:
        pass
    return False


def get_device_config() -> dict:
    """
    Get the device configuration (thread-safe, cached).

    Priority:
      1. Cached in-memory config
      2. .device_config.json from setup.py
      3. Live detection via torch
    """
    global _device_config

    with _config_lock:
        if _device_config is not None:
            return _device_config

        # Try loading from file (written by setup.py)
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    loaded = json.load(f)

                if _validate_config(loaded):
                    if _is_stale(loaded):
                        logger.warning(
                            "Device config is >%d days old. Consider running "
                            "`python setup.py --reinstall` to refresh.",
                            _STALE_DAYS,
                        )
                    _device_config = loaded
                    _device_config["_source"] = "config_file"
                    logger.info(
                        "Loaded device config: mode=%s, device=%s",
                        _device_config.get("mode"),
                        _device_config.get("training_device"),
                    )
                    return _device_config
                else:
                    logger.warning(
                        "Device config file is invalid or corrupt — using live detection"
                    )
            except json.JSONDecodeError as e:
                logger.warning("Corrupt config file %s: %s — using live detection", CONFIG_FILE, e)
            except OSError as e:
                logger.warning("Cannot read %s: %s — using live detection", CONFIG_FILE, e)

        # Fall back to live detection
        _device_config = _detect_live()
        return _device_config


def refresh_device_config() -> dict:
    """Force re-detection, bypassing the cache. Thread-safe."""
    global _device_config
    with _config_lock:
        _device_config = None
    return get_device_config()


def get_training_device() -> str:
    """
    Return the best available training device: 'cuda', 'mps', or 'cpu'.
    """
    config = get_device_config()
    return config.get("training_device", "cpu")


def get_optimal_training_defaults(device: str | None = None) -> dict:
    """
    Return optimized training defaults based on the detected device.

    This adjusts batch sizes, precision, and other settings to get the
    best performance on the available hardware.
    """
    if device is None:
        device = get_training_device()

    config = get_device_config()
    vram_gb = config.get("vram_gb")
    if vram_gb is None and config.get("gpu"):
        vram_gb = config["gpu"].get("primary_vram_gb")

    # Base defaults
    defaults = {
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.01,
    }

    if device == "cuda" and vram_gb:
        if vram_gb >= 24:
            # High-end GPU (3090, 4090, A100, etc.)
            defaults.update({
                "batch_size": 8,
                "fp16": False,
                "bf16": True,
                "max_tokens": 1024,
                "gradient_accumulation_steps": 2,
            })
        elif vram_gb >= 12:
            # Mid-range GPU (3060 12GB, 4070, etc.)
            defaults.update({
                "batch_size": 4,
                "fp16": True,
                "bf16": False,
                "max_tokens": 512,
                "gradient_accumulation_steps": 4,
            })
        elif vram_gb >= 6:
            # Low-end GPU (3060 6GB, 2060, etc.)
            defaults.update({
                "batch_size": 2,
                "fp16": True,
                "bf16": False,
                "max_tokens": 512,
                "gradient_accumulation_steps": 8,
                "gradient_checkpointing": True,
            })
        else:
            # Very low VRAM — recommend QLoRA
            defaults.update({
                "batch_size": 1,
                "fp16": True,
                "bf16": False,
                "max_tokens": 256,
                "gradient_accumulation_steps": 16,
                "gradient_checkpointing": True,
                "method": "qlora",
            })

    elif device == "mps":
        # Apple Silicon — no fp16/bf16 via the flags, PyTorch handles it
        defaults.update({
            "batch_size": 4,
            "fp16": False,
            "bf16": False,
            "max_tokens": 512,
            "gradient_accumulation_steps": 4,
        })

    else:
        # CPU — be very conservative
        defaults.update({
            "batch_size": 1,
            "fp16": False,
            "bf16": False,
            "max_tokens": 256,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "epochs": 1,
        })

    return defaults


def startup_device_check():
    """
    Run at server startup to log device info and warn about misconfigurations.
    Called from app.main lifespan.
    """
    config = get_device_config()
    device = config.get("training_device", "cpu")
    mode = config.get("mode", "cpu")

    # Log what we found
    if device == "cuda":
        gpu_name = (
            config.get("gpu_name")
            or (config.get("gpu", {}).get("primary_gpu") if config.get("gpu") else None)
            or "Unknown GPU"
        )
        vram = (
            config.get("vram_gb")
            or (config.get("gpu", {}).get("primary_vram_gb") if config.get("gpu") else None)
            or "?"
        )
        logger.info("🟢 Training device: CUDA — %s (%s GB VRAM)", gpu_name, vram)
    elif device == "mps":
        logger.info("🟢 Training device: MPS (Apple Silicon)")
    else:
        logger.info("🟡 Training device: CPU (training will be slower)")

    # Check for mismatches
    try:
        import torch
        if torch.cuda.is_available() and device == "cpu":
            logger.warning(
                "⚠️  CUDA GPU detected but running in CPU mode! "
                "Run `python setup.py` from the project root to install GPU dependencies."
            )
        elif not torch.cuda.is_available() and mode == "cuda":
            logger.warning(
                "⚠️  Setup was configured for GPU but CUDA is not available. "
                "Run `python setup.py` to reconfigure."
            )
    except ImportError:
        logger.error(
            "❌ PyTorch is not installed! Run `python setup.py` to set up dependencies."
        )

    return config
