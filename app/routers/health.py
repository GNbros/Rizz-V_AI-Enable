from fastapi import APIRouter, Depends

from app.dependencies import get_model_service, get_settings
from app.services.model_service import ModelService
from app.config import Settings

router = APIRouter()


@router.get("/")
def health(
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
):
    return {
        "status": "running",
        "model_version": model_service.version,
        "base_model": settings.base_model_name,
        "adapter": settings.adapter_path,
    }
