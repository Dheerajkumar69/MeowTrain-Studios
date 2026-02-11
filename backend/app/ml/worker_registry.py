"""
Worker registry — pure Python tracking of active TrainingWorker instances.

Separate from training_worker.py so it can be imported without torch.
"""

import logging
import threading
from typing import Optional, Any

logger = logging.getLogger("meowllm.worker_registry")

_active_workers: dict[int, Any] = {}
_workers_lock = threading.Lock()


def get_worker(project_id: int) -> Optional[Any]:
    """Get the active training worker for a project."""
    with _workers_lock:
        return _active_workers.get(project_id)


def register_worker(project_id: int, worker: Any):
    """Register a training worker."""
    with _workers_lock:
        _active_workers[project_id] = worker


def unregister_worker(project_id: int):
    """Unregister a training worker."""
    with _workers_lock:
        _active_workers.pop(project_id, None)


def cleanup_dead_workers():
    """Remove workers that are no longer running."""
    with _workers_lock:
        dead = [pid for pid, w in _active_workers.items() if not w.is_alive]
        for pid in dead:
            del _active_workers[pid]
    if dead:
        logger.info("Cleaned up %d dead workers", len(dead))
