from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_repository
from app.schemas import RatingRequest
from app.db.repository import RatingRepository, RatingEntry

router = APIRouter()


@router.post("/rating")
def rate_suggestion(
    req: RatingRequest,
    repository: RatingRepository = Depends(get_repository),
):
    if not req.suggestion:
        raise HTTPException(status_code=400, detail="Suggestion is required")
    if req.rating is not None and req.rating not in (0, 1):
        raise HTTPException(status_code=400, detail="Rating must be 0 or 1")
    if req.suggestion_type not in ("realtime", "comment-to-code"):
        raise HTTPException(status_code=400, detail="suggestion_type must be 'realtime' or 'comment-to-code'")

    entry = RatingEntry(
        prefix=req.prefix,
        suffix=req.suffix,
        suggestion=req.suggestion,
        rating=req.rating,
        suggestion_type=req.suggestion_type,
        accepted=req.accepted,
        timestamp=req.timestamp or datetime.now(timezone.utc).isoformat(),
    )
    repository.save(entry)
    return {"message": "Rating saved"}
