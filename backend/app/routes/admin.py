"""
Admin panel routes — user management, system stats, model cache.

All endpoints require ``role=admin``.

``GET    /api/admin/users``            → list all users (paginated)
``DELETE /api/admin/users/{user_id}``  → delete any user account
``PATCH  /api/admin/users/{user_id}``  → update user role
``GET    /api/admin/stats``            → system-wide statistics
``GET    /api/admin/cache``            → model cache info
``DELETE /api/admin/cache/{model_id}`` → evict a cached model
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.models.background_task import BackgroundTask
from app.services.auth_service import get_user_from_header, require_role
from app.config import PROJECTS_DIR, MODEL_CACHE_DIR

logger = logging.getLogger("meowllm.routes.admin")

router = APIRouter(prefix="/admin", tags=["Admin"])


def _require_admin(db: Session, authorization: Optional[str]) -> User:
    """Authenticate and verify the user is an admin."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    try:
        require_role(user, "admin")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return user


# ── Users ──


@router.get("/users")
def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """List all users with optional search/filter. Admin only."""
    _require_admin(db, authorization)

    query = db.query(User)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (User.email.ilike(search_term)) | (User.display_name.ilike(search_term))
        )

    if role:
        query = query.filter(User.role == role)

    total = query.count()

    # Subquery for project counts (avoids N+1)
    project_count_sq = (
        db.query(Project.user_id, func.count(Project.id).label("cnt"))
        .group_by(Project.user_id)
        .subquery()
    )
    user_ids_page = (
        query.order_by(User.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .subquery()
    )
    rows = (
        db.query(User, func.coalesce(project_count_sq.c.cnt, 0).label("project_count"))
        .outerjoin(project_count_sq, User.id == project_count_sq.c.user_id)
        .filter(User.id.in_(db.query(user_ids_page.c.id)))
        .order_by(User.id.desc())
        .all()
    )

    return {
        "items": [
            {
                "id": u.id,
                "email": u.email,
                "display_name": u.display_name,
                "is_guest": u.is_guest,
                "role": u.role,
                "email_verified": getattr(u, "email_verified", False),
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "project_count": project_count,
            }
            for u, project_count in rows
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Delete a user and all their data. Admin only."""
    admin = _require_admin(db, authorization)

    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")

    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete filesystem data
    projects = db.query(Project).filter(Project.user_id == user_id).all()
    for project in projects:
        project_dir = PROJECTS_DIR / str(project.id)
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)

    # CASCADE handles DB children
    db.delete(target)
    db.commit()

    logger.info("Admin %d deleted user %d (%s)", admin.id, user_id, target.email or "guest")
    return {"detail": f"User {user_id} and all associated data deleted"}


@router.patch("/users/{user_id}")
def update_user_role(
    user_id: int,
    role: str = Query(..., pattern="^(admin|member|guest)$"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Change a user's role. Admin only."""
    admin = _require_admin(db, authorization)

    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    old_role = target.role
    target.role = role
    target.is_guest = (role == "guest")
    db.commit()

    logger.info("Admin %d changed user %d role: %s → %s", admin.id, user_id, old_role, role)
    return {"detail": f"User {user_id} role changed from {old_role} to {role}"}


# ── System Stats ──


def _dir_size_gb(path: Path) -> float:
    """Calculate total size of a directory in GB."""
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / (1024 ** 3), 3)
    except Exception:
        return 0.0


@router.get("/stats")
def system_stats(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """System-wide statistics. Admin only."""
    _require_admin(db, authorization)

    total_users = db.query(func.count(User.id)).scalar() or 0
    guest_users = db.query(func.count(User.id)).filter(User.is_guest == True).scalar() or 0  # noqa: E712
    total_projects = db.query(func.count(Project.id)).scalar() or 0
    total_datasets = db.query(func.count(Dataset.id)).scalar() or 0
    total_training_runs = db.query(func.count(TrainingRun.id)).scalar() or 0
    active_runs = (
        db.query(func.count(TrainingRun.id))
        .filter(TrainingRun.status.in_(("running", "initializing", "paused")))
        .scalar() or 0
    )
    completed_runs = (
        db.query(func.count(TrainingRun.id))
        .filter(TrainingRun.status == "completed")
        .scalar() or 0
    )
    error_runs = (
        db.query(func.count(TrainingRun.id))
        .filter(TrainingRun.status == "error")
        .scalar() or 0
    )
    total_model_configs = db.query(func.count(ModelConfig.id)).scalar() or 0
    active_tasks = (
        db.query(func.count(BackgroundTask.id))
        .filter(BackgroundTask.status == "running")
        .scalar() or 0
    )

    return {
        "users": {
            "total": total_users,
            "guests": guest_users,
            "registered": total_users - guest_users,
        },
        "projects": {
            "total": total_projects,
        },
        "datasets": {
            "total": total_datasets,
        },
        "training": {
            "total_runs": total_training_runs,
            "active": active_runs,
            "completed": completed_runs,
            "errors": error_runs,
        },
        "models": {
            "configs": total_model_configs,
        },
        "tasks": {
            "active": active_tasks,
        },
        "disk": {
            "projects_gb": _dir_size_gb(PROJECTS_DIR),
            "model_cache_gb": _dir_size_gb(MODEL_CACHE_DIR),
        },
    }


# ── Model Cache Management ──


@router.get("/cache")
def list_cache(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """List all cached models with sizes. Admin only."""
    _require_admin(db, authorization)

    entries = []
    if MODEL_CACHE_DIR.exists():
        for child in sorted(MODEL_CACHE_DIR.iterdir()):
            if child.is_dir():
                size_gb = _dir_size_gb(child)
                entries.append({
                    "name": child.name,
                    "size_gb": size_gb,
                    "path": str(child),
                })

    return {
        "total_size_gb": round(sum(e["size_gb"] for e in entries), 3),
        "entries": entries,
    }


@router.delete("/cache/{model_name:path}")
def evict_cache(
    model_name: str,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Delete a cached model from disk. Admin only."""
    _require_admin(db, authorization)

    # Prevent path traversal
    if ".." in model_name or model_name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid model name")

    target = MODEL_CACHE_DIR / model_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Cache entry not found")

    if not str(target.resolve()).startswith(str(MODEL_CACHE_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")

    size_before = _dir_size_gb(target)
    shutil.rmtree(target, ignore_errors=True)

    logger.info("Admin evicted cache entry '%s' (%.2f GB)", model_name, size_before)
    return {"detail": f"Cache entry '{model_name}' deleted ({size_before:.2f} GB freed)"}
