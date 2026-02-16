"""
Lightweight background task tracking.

Stores the state of long-running operations (downloads, GGUF conversions,
augmentation) in the DB so that server restarts don't silently lose
in-progress work.  On startup, any task stuck in "running" is marked
as "interrupted" so the user knows to retry.

This is NOT a full task queue (Celery/ARQ/Dramatiq).  If you need
reliable distributed workers, add one of those and migrate.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime, timezone
from app.database import Base


class BackgroundTask(Base):
    __tablename__ = "background_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(String, nullable=False, index=True)   # download, gguf, augment
    task_key = Column(String, nullable=False, index=True)    # e.g. model_id or project_id
    status = Column(String, default="running")               # running, completed, error, interrupted
    progress = Column(Float, default=0.0)
    message = Column(String, default="")
    error = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)      # extra task-specific data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
