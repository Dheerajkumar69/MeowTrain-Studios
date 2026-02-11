from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional
import shutil

from app.database import get_db
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.model_config import ModelConfig
from app.models.training_run import TrainingRun
from app.models.prompt_template import PromptTemplate
from app.schemas import ProjectCreate, ProjectUpdate, ProjectResponse
from app.services.auth_service import get_user_from_header
from app.services.hardware_service import get_hardware_status
from app.config import PROJECTS_DIR

router = APIRouter(prefix="/projects", tags=["Projects"])


def _get_user(authorization: Optional[str], db: Session):
    try:
        return get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


def _project_to_response(project: Project, db: Session) -> dict:
    dataset_count = db.query(Dataset).filter(Dataset.project_id == project.id).count()
    config_count = db.query(ModelConfig).filter(ModelConfig.project_id == project.id).count()
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


@router.get("/", response_model=list[ProjectResponse])
def list_projects(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = _get_user(authorization, db)
    projects = db.query(Project).filter(Project.user_id == user.id).order_by(Project.updated_at.desc()).all()
    return [_project_to_response(p, db) for p in projects]


@router.post("/", response_model=ProjectResponse)
def create_project(req: ProjectCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = _get_user(authorization, db)
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

    # Delete filesystem
    project_dir = PROJECTS_DIR / str(project_id)
    if project_dir.exists():
        shutil.rmtree(project_dir)

    # Cascade delete related DB records
    db.query(PromptTemplate).filter(PromptTemplate.project_id == project.id).delete()
    db.query(TrainingRun).filter(TrainingRun.project_id == project.id).delete()
    db.query(ModelConfig).filter(ModelConfig.project_id == project.id).delete()
    db.query(Dataset).filter(Dataset.project_id == project.id).delete()

    db.delete(project)
    db.commit()
    return {"detail": "Project deleted"}
