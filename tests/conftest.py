"""
conftest.py

Stubs out the ML stack so tests run without model files or a GPU.
Uses FastAPI dependency_overrides — no monkeypatching of module globals.
"""

import sys
from unittest.mock import MagicMock
from typing import Optional

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock heavy ML deps BEFORE any app code is imported
# ---------------------------------------------------------------------------
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
_torch.backends.mps.is_available.return_value = False
_torch.float32 = "float32"
_torch.float16 = "float16"
_no_grad = MagicMock()
_no_grad.__enter__ = MagicMock(return_value=None)
_no_grad.__exit__ = MagicMock(return_value=False)
_torch.no_grad.return_value = _no_grad

sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.backends", _torch.backends)
sys.modules.setdefault("torch.backends.mps", _torch.backends.mps)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("peft", MagicMock())

# ---------------------------------------------------------------------------
# Import app code after mocks are in place
# ---------------------------------------------------------------------------
from app.main import create_app                          # noqa: E402
from app.config import Settings                          # noqa: E402
from app.dependencies import get_model_service, get_repository  # noqa: E402
from app.services.model_service import ModelService      # noqa: E402
from app.db.repository import RatingRepository, RatingEntry  # noqa: E402


# ---------------------------------------------------------------------------
# Fake services
# ---------------------------------------------------------------------------

class FakeModelService:
    """Returns predictable completions — no model files needed."""
    version = "test-v1"

    def __init__(self):
        self.settings = Settings(
            base_model_name="fake-model",
            adapter_path="fake-adapter",
            model_version="test-v1",
        )

    def load(self) -> None:
        pass

    def complete(self, prefix: str, suffix: str, max_new_tokens: int) -> str:
        return "addi t0, t0, 1"


class FakeRatingRepository:
    """In-memory store — no SQLite file needed."""

    def __init__(self):
        self._store: list[RatingEntry] = []

    def init_db(self) -> None:
        pass

    def save(self, entry: RatingEntry) -> None:
        self._store.append(entry)

    def find_all(self) -> list[RatingEntry]:
        return list(self._store)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_repository():
    return FakeRatingRepository()


@pytest.fixture
def app(fake_repository):
    application = create_app(settings=Settings(db_path=":memory:"))
    application.dependency_overrides[get_model_service] = lambda: FakeModelService()
    application.dependency_overrides[get_repository] = lambda: fake_repository
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
