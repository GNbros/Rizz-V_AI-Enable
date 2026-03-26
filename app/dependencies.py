from fastapi import Request

from app.config import Settings
from app.services.model_service import ModelService
from app.db.repository import RatingRepository


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_repository(request: Request) -> RatingRepository:
    return request.app.state.repository
