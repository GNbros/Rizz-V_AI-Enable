from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.dependencies import get_model_service, get_settings
from app.schemas import PromptRequest, PromptResponse
from app.services.model_service import ModelService
from app.config import Settings

router = APIRouter()


@router.post("/generate", response_model=PromptResponse)
async def generate_code(
    req: PromptRequest,
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_settings),
):
    if not req.prefix:
        raise HTTPException(status_code=400, detail="Prefix is required")
    if req.max_new_tokens <= 0:
        raise HTTPException(status_code=400, detail="max_new_tokens must be greater than 0")
    if len(req.prefix) > settings.max_input_length:
        raise HTTPException(status_code=400, detail=f"Prefix exceeds {settings.max_input_length} characters")

    # Run blocking inference in thread pool so the event loop stays free
    result = await run_in_threadpool(
        model_service.complete, req.prefix, req.suffix, req.max_new_tokens
    )
    return {"generated_code": result}
