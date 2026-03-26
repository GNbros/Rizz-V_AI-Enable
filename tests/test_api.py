"""
test_api.py — unit tests for all API endpoints and DB operations.

Run with:
    pytest tests/ -v
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

from app.dependencies import get_model_service
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_root_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_root_message(self, client):
        data = client.get("/").json()
        assert data["status"] == "running"
        assert "model_version" in data


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_basic_suggestion_returned(self, client):
        res = client.post("/generate", json={"prefix": "addi t0, t0,", "suffix": ""})
        assert res.status_code == 200
        assert "generated_code" in res.json()

    def test_fim_includes_suffix(self):
        """ModelService.complete() must build a FIM prompt containing both prefix and suffix."""
        from app.services.model_service import ModelService
        from app.config import Settings

        settings = Settings(base_model_name="fake", adapter_path="fake")
        svc = ModelService(settings)

        captured = {}

        def capture_tokenize(prompt, **kwargs):
            captured["prompt"] = prompt
            return {"input_ids": MagicMock()}

        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = capture_tokenize
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = "result"

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()

        svc._tokenizer = mock_tokenizer
        svc._model = mock_model

        svc.complete("# multiply\naddi t0,", "\nret", 30)

        prompt = captured["prompt"]
        assert "<fim_prefix>" in prompt
        assert "<fim_suffix>" in prompt
        assert "<fim_middle>" in prompt
        assert "# multiply" in prompt
        assert "ret" in prompt

    def test_empty_prefix_returns_400(self, client):
        res = client.post("/generate", json={"prefix": "", "suffix": ""})
        assert res.status_code == 400
        assert "detail" in res.json()

    def test_prefix_too_long_returns_400(self, client):
        res = client.post("/generate", json={"prefix": "a" * 513, "suffix": ""})
        assert res.status_code == 400
        assert "detail" in res.json()

    def test_zero_max_new_tokens_returns_400(self, client):
        res = client.post("/generate", json={"prefix": "addi", "suffix": "", "max_new_tokens": 0})
        assert res.status_code == 400
        assert "detail" in res.json()

    def test_missing_prefix_field_returns_422(self, client):
        res = client.post("/generate", json={"suffix": "ret"})
        assert res.status_code == 422

    def test_custom_max_tokens_passed_through(self, app, fake_repository):
        captured = {}

        class SpyModelService:
            version = "test-v1"

            def load(self):
                pass

            def complete(self, prefix, suffix, max_new_tokens):
                captured["max_new_tokens"] = max_new_tokens
                return "nop"

        app.dependency_overrides[get_model_service] = lambda: SpyModelService()

        with TestClient(app) as c:
            c.post("/generate", json={"prefix": "addi", "suffix": "", "max_new_tokens": 120})

        assert captured["max_new_tokens"] == 120


# ---------------------------------------------------------------------------
# POST /rating
# ---------------------------------------------------------------------------

class TestRating:
    _base = {
        "prefix": "addi t0, t0,",
        "suffix": "\nret",
        "suggestion": "addi t0, t0, 1",
        "suggestion_type": "realtime",
        "accepted": True,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }

    def test_helpful_rating_saved(self, client, fake_repository):
        res = client.post("/rating", json={**self._base, "rating": 1})
        assert res.status_code == 200
        assert res.json()["message"] == "Rating saved"
        assert len(fake_repository._store) == 1
        assert fake_repository._store[0].rating == 1

    def test_unhelpful_rating_saved(self, client, fake_repository):
        res = client.post("/rating", json={**self._base, "rating": 0})
        assert res.status_code == 200
        assert fake_repository._store[0].rating == 0

    def test_null_rating_allowed(self, client, fake_repository):
        """User closed the popup without rating — rating should be None."""
        res = client.post("/rating", json={**self._base, "rating": None})
        assert res.status_code == 200
        assert fake_repository._store[0].rating is None

    def test_invalid_rating_value_returns_400(self, client):
        res = client.post("/rating", json={**self._base, "rating": 3})
        assert res.status_code == 400
        assert "detail" in res.json()

    def test_empty_suggestion_returns_400(self, client):
        res = client.post("/rating", json={**self._base, "suggestion": ""})
        assert res.status_code == 400
        assert "detail" in res.json()

    def test_invalid_suggestion_type_returns_400(self, client):
        res = client.post("/rating", json={**self._base, "suggestion_type": "unknown"})
        assert res.status_code == 400
        assert "detail" in res.json()

    def test_comment_to_code_type_accepted(self, client):
        res = client.post("/rating", json={**self._base, "suggestion_type": "comment-to-code", "rating": 1})
        assert res.status_code == 200
        assert "detail" not in res.json()

    def test_prefix_and_suffix_stored_in_db(self, client, fake_repository):
        client.post("/rating", json={**self._base, "rating": 1})
        entry = fake_repository._store[0]
        assert entry.prefix == self._base["prefix"]
        assert entry.suffix == self._base["suffix"]

    def test_missing_prefix_field_returns_422(self, client):
        payload = {k: v for k, v in self._base.items() if k != "prefix"}
        res = client.post("/rating", json=payload)
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Database (RatingRepository unit tests)
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_db_creates_table(self, tmp_path):
        from app.db.repository import RatingRepository

        repo = RatingRepository(str(tmp_path / "fresh.db"))
        repo.init_db()

        conn = sqlite3.connect(repo.db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        assert ("rating",) in tables

    def test_migration_adds_missing_columns(self, tmp_path):
        """init_db should add new columns to an old-schema DB without crashing."""
        from app.db.repository import RatingRepository

        db_path = str(tmp_path / "old.db")
        # Simulate an older schema that is missing several columns
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE rating (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prefix TEXT NOT NULL,
                suggestion TEXT NOT NULL,
                rating INTEGER
            )
        """)
        conn.commit()
        conn.close()

        RatingRepository(db_path).init_db()  # should not raise

        conn = sqlite3.connect(db_path)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(rating)").fetchall()]
        conn.close()
        assert "suffix" in cols
        assert "suggestion_type" in cols
        assert "accepted" in cols
        assert "timestamp" in cols
