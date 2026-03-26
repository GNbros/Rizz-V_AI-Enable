"""
conftest.py — shared fixtures and model mocking.

The ML stack (torch / transformers / peft) is replaced with MagicMocks
before `main` is imported, so no trained-model files are needed to run tests.
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1. Stub out the ML dependencies BEFORE importing main
# ---------------------------------------------------------------------------

_torch = MagicMock()
_torch.cuda.is_available.return_value = False
_torch.backends.mps.is_available.return_value = False
_torch.float32 = "float32"
_torch.float16 = "float16"

# Make `with torch.no_grad():` work as a context manager
_no_grad_ctx = MagicMock()
_no_grad_ctx.__enter__ = MagicMock(return_value=None)
_no_grad_ctx.__exit__ = MagicMock(return_value=False)
_torch.no_grad.return_value = _no_grad_ctx

sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.backends", _torch.backends)
sys.modules.setdefault("torch.backends.mps", _torch.backends.mps)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("peft", MagicMock())

# ---------------------------------------------------------------------------
# 2. Now import main (load_model runs but uses mocked objects)
# ---------------------------------------------------------------------------

import main as _main  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# 3. Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    return TestClient(_main.app)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Each test gets a fresh SQLite database — never touches ratings.db."""
    db_path = str(tmp_path / "test_ratings.db")
    monkeypatch.setattr(_main, "DB_PATH", db_path)
    _main.init_db()


@pytest.fixture(autouse=True)
def mock_inference(monkeypatch):
    """Replace complete_code with a fast, predictable stub."""
    def _fake_complete(model, tokenizer, prompt, max_new_tokens=50):
        return "addi t0, t0, 1"

    monkeypatch.setattr(_main, "complete_code", _fake_complete)
