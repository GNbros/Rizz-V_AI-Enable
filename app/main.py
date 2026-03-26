from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings
from app.services.model_service import ModelService
from app.db.repository import RatingRepository
from app.routers import health, generate, rating


def create_app(settings: Settings | None = None) -> FastAPI:
    _settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Store settings on app state so dependencies can read them
        app.state.settings = _settings

        # Init DB
        app.state.repository = RatingRepository(_settings.db_path)
        app.state.repository.init_db()

        # Load model — only touches files/GPU here, nowhere else
        app.state.model_service = ModelService(_settings)
        app.state.model_service.load()

        yield
        # shutdown: nothing to clean up for SQLite or the model

    app = FastAPI(
        title="Rizz-V Code Generation API",
        version="1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(rating.router)

    return app
