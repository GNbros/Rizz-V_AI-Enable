from pydantic import BaseModel
from typing import Optional


class PromptRequest(BaseModel):
    prefix: str
    suffix: str = ""
    max_new_tokens: int = 30


class PromptResponse(BaseModel):
    generated_code: str


class RatingRequest(BaseModel):
    prefix: str
    suffix: str = ""
    suggestion: str
    rating: Optional[int] = None
    suggestion_type: str = "realtime"
    accepted: bool = True
    timestamp: str = ""
