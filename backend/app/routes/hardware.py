from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import Optional

from app.schemas import HardwareStatus
from app.services.hardware_service import get_hardware_status
from app.services.auth_service import get_user_from_header
from app.database import get_db

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("/", response_model=HardwareStatus)
def hardware_status(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    hw = get_hardware_status()
    return HardwareStatus(**hw)


@router.get("/device")
def device_info(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Return the detected training device configuration.

    Includes what device is available (cuda/mps/cpu), GPU info,
    and recommended training defaults optimized for this hardware.
    """
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    from app.device_config import get_device_config, get_optimal_training_defaults

    config = get_device_config()
    defaults = get_optimal_training_defaults()

    return {
        "training_device": config.get("training_device", "cpu"),
        "mode": config.get("mode", "cpu"),
        "gpu": config.get("gpu"),
        "gpu_name": config.get("gpu_name"),
        "vram_gb": config.get("vram_gb"),
        "cuda_version": config.get("cuda_version"),
        "apple_silicon": config.get("apple_silicon", False),
        "torch_version": config.get("torch_version")
            or config.get("validation", {}).get("torch_version"),
        "recommended_defaults": defaults,
        # PRIME / Optimus hints (Linux laptops with hybrid graphics)
        "prime_blocked": config.get("_prime_blocked", False),
        "prime_mode": config.get("_prime_mode"),
    }


@router.post("/refresh-device")
def refresh_device(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Force re-detection of hardware. Clears the cached config and
    re-detects via torch. Useful after driver updates or GPU changes.
    """
    try:
        get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    from app.device_config import refresh_device_config, get_optimal_training_defaults

    config = refresh_device_config()
    defaults = get_optimal_training_defaults()

    return {
        "training_device": config.get("training_device", "cpu"),
        "mode": config.get("mode", "cpu"),
        "gpu_name": config.get("gpu_name"),
        "vram_gb": config.get("vram_gb"),
        "recommended_defaults": defaults,
        "refreshed": True,
    }
