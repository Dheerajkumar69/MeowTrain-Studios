"""
ML module for MeowLLM training pipeline.

Components:
  - data_loader: Dataset loading, formatting, tokenization
  - trainer: HuggingFace Trainer wrapper with real-time metrics
  - checkpoint_manager: Checkpoint saving, pruning, LoRA merge
  - training_worker: Background thread worker for training lifecycle
  - worker_registry: Pure Python worker tracking (no torch dependency)

Note: Heavy imports (torch, transformers) are deferred until runtime
to allow the app to start and tests to run without GPU dependencies.
"""

# Registry functions — pure Python, always available
from app.ml.worker_registry import (
    get_worker,
    register_worker,
    unregister_worker,
    cleanup_dead_workers,
)


def _get_training_worker_class():
    """Lazy import to avoid loading torch at import time."""
    from app.ml.training_worker import TrainingWorker
    return TrainingWorker


def _get_training_metrics_class():
    """Lazy import to avoid loading torch at import time."""
    from app.ml.trainer import TrainingMetrics
    return TrainingMetrics


# Lazy factory for TrainingWorker
class TrainingWorker:
    """Proxy that lazily imports the real TrainingWorker."""
    def __new__(cls, *args, **kwargs):
        real_cls = _get_training_worker_class()
        return real_cls(*args, **kwargs)


class TrainingMetrics:
    """Proxy that lazily imports the real TrainingMetrics."""
    def __new__(cls, *args, **kwargs):
        real_cls = _get_training_metrics_class()
        return real_cls(*args, **kwargs)


__all__ = [
    "TrainingWorker",
    "TrainingMetrics",
    "get_worker",
    "register_worker",
    "unregister_worker",
    "cleanup_dead_workers",
]
