from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.schemas import LMStudioConfigResponse, LMStudioConnectionResponse, LMStudioModelsResponse
from app.services.auth_service import get_user_from_header
from app.services.lmstudio_service import (
    get_lmstudio_config,
    update_lmstudio_config,
    check_connection,
    list_models,
)

router = APIRouter(prefix="/lmstudio", tags=["LM Studio"])


class LMStudioConfigRequest(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    enabled: Optional[bool] = None


@router.get("/config", response_model=LMStudioConfigResponse)
def get_config(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Get current LM Studio connection settings."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    config = get_lmstudio_config()
    return config


@router.put("/config", response_model=LMStudioConfigResponse)
def set_config(
    req: LMStudioConfigRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Update LM Studio connection settings."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if user.is_guest:
        raise HTTPException(status_code=403, detail="Guest users cannot modify server settings")

    try:
        update_lmstudio_config(host=req.host, port=req.port, enabled=req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return get_lmstudio_config()


@router.post("/test", response_model=LMStudioConnectionResponse)
def test_connection(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Test if LM Studio server is reachable and return status."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    result = check_connection()
    if result["connected"]:
        models = list_models()
        result["models"] = models
    return result


@router.get("/models", response_model=LMStudioModelsResponse)
def get_lmstudio_models(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """List models currently loaded in LM Studio."""
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    config = get_lmstudio_config()
    if not config["enabled"]:
        return {"models": [], "enabled": False, "message": "LM Studio integration is disabled"}

    connection = check_connection()
    if not connection["connected"]:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to LM Studio: {connection.get('error', 'Unknown error')}. "
                   f"Make sure LM Studio is running with the local server enabled."
        )

    models = list_models()
    return {
        "models": models,
        "enabled": True,
        "url": connection["url"],
    }
