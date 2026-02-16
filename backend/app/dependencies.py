"""
Shared FastAPI dependencies.

Reusable helpers that eliminate the duplicated ``_get_project`` pattern
across route modules.
"""

from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.project import Project
from app.services.auth_service import get_user_from_header


def get_project_for_user(
    project_id: int,
    authorization: Optional[str],
    db: Session,
) -> Project:
    """Authenticate the caller and return the requested project.

    Raises:
        HTTPException 401 – invalid / missing token
        HTTPException 404 – project does not exist or is not owned by user
    """
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.user_id == user.id)
        .first()
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
