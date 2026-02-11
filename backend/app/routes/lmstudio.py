from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

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


@router.get("/config")
def get_config():
    """Get current LM Studio connection settings."""
    config = get_lmstudio_config()
    return config


@router.put("/config")
def set_config(req: LMStudioConfigRequest):
    """Update LM Studio connection settings."""
    update_lmstudio_config(host=req.host, port=req.port, enabled=req.enabled)
    return get_lmstudio_config()


@router.post("/test")
def test_connection():
    """Test if LM Studio server is reachable and return status."""
    result = check_connection()
    if result["connected"]:
        models = list_models()
        result["models"] = models
    return result


@router.get("/models")
def get_lmstudio_models():
    """List models currently loaded in LM Studio."""
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
