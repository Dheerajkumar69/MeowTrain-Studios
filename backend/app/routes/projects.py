from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Optional
import shutil

from app.database import get_db
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.models.prompt_template import PromptTemplate
from app.schemas import ProjectCreate, ProjectUpdate, ProjectResponse, PaginatedProjectsResponse
from app.services.auth_service import get_user_from_header
from app.services.hardware_service import get_hardware_status
from app.config import PROJECTS_DIR, GUEST_MAX_PROJECTS

router = APIRouter(prefix="/projects", tags=["Projects"])


def _get_user(authorization: Optional[str], db: Session):
    try:
        return get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


def _project_to_response(project: Project, db: Session) -> dict:
    """Convert a single project to response (for create/get/update)."""
    dataset_count = db.query(func.count(Dataset.id)).filter(Dataset.project_id == project.id).scalar()
    config_count = db.query(func.count(ModelConfig.id)).filter(ModelConfig.project_id == project.id).scalar()
    return ProjectResponse(
        id=project.id,
        user_id=project.user_id,
        name=project.name,
        description=project.description,
        intended_use=project.intended_use,
        status=project.status,
        hardware_snapshot=project.hardware_snapshot,
        created_at=project.created_at,
        updated_at=project.updated_at,
        dataset_count=dataset_count,
        model_config_count=config_count,
    )


@router.get("/", response_model=PaginatedProjectsResponse)
def list_projects(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
    page: int = 1,
    per_page: int = 50,
    search: Optional[str] = None,
    status: Optional[str] = None,
):
    user = _get_user(authorization, db)
    per_page = min(max(per_page, 1), 100)
    page = max(page, 1)
    offset = (page - 1) * per_page

    # ── Subqueries for counts (avoids N+1: one query total instead of 2N+1) ──
    dataset_count_sq = (
        db.query(Dataset.project_id, func.count(Dataset.id).label("cnt"))
        .group_by(Dataset.project_id)
        .subquery()
    )
    config_count_sq = (
        db.query(ModelConfig.project_id, func.count(ModelConfig.id).label("cnt"))
        .group_by(ModelConfig.project_id)
        .subquery()
    )

    base_query = (
        db.query(
            Project,
            func.coalesce(dataset_count_sq.c.cnt, 0).label("dataset_count"),
            func.coalesce(config_count_sq.c.cnt, 0).label("model_config_count"),
        )
        .outerjoin(dataset_count_sq, Project.id == dataset_count_sq.c.project_id)
        .outerjoin(config_count_sq, Project.id == config_count_sq.c.project_id)
        .filter(Project.user_id == user.id)
    )

    # ── Search by name ──
    if search:
        # Escape SQL LIKE special characters to prevent pattern injection
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        base_query = base_query.filter(Project.name.ilike(f"%{escaped}%", escape="\\"))

    # ── Filter by status ──
    if status:
        base_query = base_query.filter(Project.status == status)

    total = base_query.count()

    rows = (
        base_query
        .order_by(Project.updated_at.desc())
        .offset(offset)
        .limit(per_page)
        .all()
    )

    items = [
        ProjectResponse(
            id=proj.id,
            user_id=proj.user_id,
            name=proj.name,
            description=proj.description,
            intended_use=proj.intended_use,
            status=proj.status,
            hardware_snapshot=proj.hardware_snapshot,
            created_at=proj.created_at,
            updated_at=proj.updated_at,
            dataset_count=ds_count,
            model_config_count=mc_count,
        )
        for proj, ds_count, mc_count in rows
    ]

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.post("/", response_model=ProjectResponse)
def create_project(req: ProjectCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = _get_user(authorization, db)

    # Guest users have a project limit
    if user.is_guest:
        existing_count = db.query(Project).filter(Project.user_id == user.id).count()
        if existing_count >= GUEST_MAX_PROJECTS:
            raise HTTPException(
                status_code=403,
                detail=f"Guest accounts are limited to {GUEST_MAX_PROJECTS} projects. "
                       "Please register for an unlimited account.",
            )

    hw = get_hardware_status()

    project = Project(
        user_id=user.id,
        name=req.name,
        description=req.description,
        intended_use=req.intended_use,
        hardware_snapshot=hw,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Create filesystem dirs
    project_dir = PROJECTS_DIR / str(project.id)
    (project_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (project_dir / "processed").mkdir(parents=True, exist_ok=True)
    (project_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (project_dir / "adapters").mkdir(parents=True, exist_ok=True)

    return _project_to_response(project, db)


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = _get_user(authorization, db)
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_to_response(project, db)


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, req: ProjectUpdate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = _get_user(authorization, db)
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(project, key, value)

    db.commit()
    db.refresh(project)
    return _project_to_response(project, db)


@router.delete("/{project_id}")
def delete_project(project_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = _get_user(authorization, db)
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Unload inference model from GPU if cached for this project
    try:
        from app.services.inference_service import _get_model_path_for_project, unload_model
        model_path = _get_model_path_for_project(project_id)
        if model_path:
            unload_model(model_path)
    except Exception:
        pass  # Best-effort — don't block deletion

    # Stop any active training worker
    try:
        from app.ml.worker_registry import get_worker, unregister_worker
        worker = get_worker(project_id)
        if worker and worker.is_alive:
            worker.stop()
        unregister_worker(project_id)
    except Exception:
        pass  # Best-effort

    # Delete filesystem (verify path stays within PROJECTS_DIR to prevent traversal)
    project_dir = (PROJECTS_DIR / str(project_id)).resolve()
    projects_base = PROJECTS_DIR.resolve()
    if str(project_dir).startswith(str(projects_base)) and project_dir.exists():
        # Clean up GGUF exports, zip exports, checkpoints — entire tree
        shutil.rmtree(project_dir)

    # Cascade delete related DB records
    db.query(PromptTemplate).filter(PromptTemplate.project_id == project.id).delete()
    db.query(TrainingRun).filter(TrainingRun.project_id == project.id).delete()
    db.query(ModelConfig).filter(ModelConfig.project_id == project.id).delete()
    db.query(Dataset).filter(Dataset.project_id == project.id).delete()

    # Clean up any background tasks for this project
    from app.models.background_task import BackgroundTask
    db.query(BackgroundTask).filter(
        BackgroundTask.task_key == str(project.id),
        BackgroundTask.task_type.in_(("gguf", "augment")),
    ).delete(synchronize_session=False)

    db.delete(project)
    db.commit()
    return {"detail": "Project deleted"}
