from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional
import time

from app.database import get_db
from app.models.project import Project
from app.models.training_run import TrainingRun
from app.models.prompt_template import PromptTemplate
from app.schemas import ChatRequest, ChatResponse, PromptTemplateCreate, PromptTemplateResponse
from app.services.auth_service import get_user_from_header

router = APIRouter(prefix="/projects/{project_id}", tags=["Inference"])


def _get_project(project_id: int, authorization: Optional[str], db: Session):
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/chat", response_model=ChatResponse)
def chat(
    project_id: int,
    req: ChatRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)

    start_time = time.time()

    # Path 1: LM Studio model (explicitly requested)
    if req.lmstudio_model:
        from app.services.lmstudio_service import chat_completion, get_lmstudio_config

        config = get_lmstudio_config()
        if not config["enabled"]:
            raise HTTPException(status_code=400, detail="LM Studio integration is not enabled. Enable it in Settings.")

        try:
            messages = []
            if req.system_prompt:
                messages.append({"role": "system", "content": req.system_prompt})
            messages.append({"role": "user", "content": req.prompt})

            result = chat_completion(
                model_id=req.lmstudio_model,
                messages=messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

            elapsed_ms = (time.time() - start_time) * 1000
            return ChatResponse(
                response=result["response"],
                tokens_used=result["tokens_used"],
                generation_time_ms=round(elapsed_ms, 2),
                model_used=result.get("model", req.lmstudio_model),
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LM Studio error: {str(e)}")

    # Path 2: Native inference from fine-tuned model
    try:
        from app.services.inference_service import generate_response

        result = generate_response(
            project_id=project.id,
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        return ChatResponse(
            response=result["response"],
            tokens_used=result["tokens_used"],
            generation_time_ms=result["generation_time_ms"],
            model_used=f"fine-tuned ({result.get('model_path', 'local')})",
        )
    except RuntimeError as e:
        error_msg = str(e)
        if "No fine-tuned model" in error_msg:
            # No trained model — tell the user what to do
            raise HTTPException(
                status_code=400,
                detail=(
                    "No model available for inference. Either:\n"
                    "1. Train a model first using the Training tab\n"
                    "2. Connect LM Studio and select a model"
                ),
            )
        raise HTTPException(status_code=500, detail=f"Inference error: {error_msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/chat/context")
def get_context(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)
    from app.models.dataset import Dataset
    datasets = db.query(Dataset).filter(Dataset.project_id == project.id, Dataset.status == "ready").all()

    # Also include model info
    model_info = None
    try:
        from app.services.inference_service import get_model_info
        model_info = get_model_info(project.id)
    except Exception:
        pass

    return {
        "datasets": [
            {"id": d.id, "name": d.original_name, "tokens": d.token_count}
            for d in datasets
        ],
        "model": model_info,
    }


@router.post("/prompts", response_model=PromptTemplateResponse)
def save_prompt(
    project_id: int,
    req: PromptTemplateCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)
    template = PromptTemplate(
        project_id=project.id,
        name=req.name,
        system_prompt=req.system_prompt,
        user_prompt=req.user_prompt,
        temperature=req.temperature,
    )
    db.add(template)
    db.commit()
    db.refresh(template)
    return PromptTemplateResponse.model_validate(template)


@router.get("/prompts", response_model=list[PromptTemplateResponse])
def list_prompts(
    project_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    project = _get_project(project_id, authorization, db)
    templates = db.query(PromptTemplate).filter(PromptTemplate.project_id == project.id).all()
    return [PromptTemplateResponse.model_validate(t) for t in templates]
