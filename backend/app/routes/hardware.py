from fastapi import APIRouter
from app.schemas import HardwareStatus
from app.services.hardware_service import get_hardware_status

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("/", response_model=HardwareStatus)
def hardware_status():
    hw = get_hardware_status()
    return HardwareStatus(**hw)
