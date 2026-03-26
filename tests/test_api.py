"""
test_api.py — unit tests for all API endpoints and DB operations.

Run with:
    pytest tests/ -v
"""

import sqlite3
import sys
from unittest.mock import MagicMock

import pytest

import main as _main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_ratings(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM rating").fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# GET /  — health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_root_returns_200(self, client):
        res = client.get("/")
        assert res.status_code == 200

    def test_root_message(self, client):
        res = client.get("/")
        assert "running" in res.json()["message"].lower()


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_basic_suggestion_returned(self, client):
        res = client.post("/generate", json={"prefix": "addi t0, t0,", "suffix": ""})
        assert res.status_code == 200
        assert "generated_code" in res.json()

    def test_fim_includes_suffix(self, client, monkeypatch):
        """complete_code must receive a prompt that contains both prefix and suffix."""
        captured = {}

        def _capture(model, tokenizer, prompt, max_new_tokens=50):
            captured["prompt"] = prompt
            return "addi t0, t0, 1"

        monkeypatch.setattr(_main, "complete_code", _capture)

        client.post("/generate", json={
            "prefix": "# multiply\naddi t0,",
            "suffix": "\nret",
        })

        assert "<fim_prefix>" in captured["prompt"]
        assert "<fim_suffix>" in captured["prompt"]
        assert "<fim_middle>" in captured["prompt"]
        assert "# multiply" in captured["prompt"]
        assert "ret" in captured["prompt"]

    def test_empty_prefix_returns_error(self, client):
        res = client.post("/generate", json={"prefix": "", "suffix": ""})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_prefix_too_long_returns_error(self, client):
        res = client.post("/generate", json={"prefix": "a" * 513, "suffix": ""})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_zero_max_new_tokens_returns_error(self, client):
        res = client.post("/generate", json={"prefix": "addi", "suffix": "", "max_new_tokens": 0})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_missing_prefix_field_returns_422(self, client):
        res = client.post("/generate", json={"suffix": "ret"})
        assert res.status_code == 422

    def test_custom_max_tokens_passed_through(self, client, monkeypatch):
        captured = {}

        def _capture(model, tokenizer, prompt, max_new_tokens=50):
            captured["max_new_tokens"] = max_new_tokens
            return "nop"

        monkeypatch.setattr(_main, "complete_code", _capture)
        client.post("/generate", json={"prefix": "addi", "suffix": "", "max_new_tokens": 120})
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

    def test_helpful_rating_saved(self, client, isolated_db, monkeypatch, tmp_path):
        db_path = str(tmp_path / "test_ratings.db")
        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()

        res = client.post("/rating", json={**self._base, "rating": 1})
        assert res.status_code == 200
        assert res.json()["message"] == "Rating saved"

        rows = _query_ratings(db_path)
        assert len(rows) == 1
        assert rows[0][4] == 1  # rating column

    def test_unhelpful_rating_saved(self, client, isolated_db, monkeypatch, tmp_path):
        db_path = str(tmp_path / "test_ratings.db")
        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()

        res = client.post("/rating", json={**self._base, "rating": 0})
        assert res.status_code == 200
        rows = _query_ratings(db_path)
        assert rows[0][4] == 0

    def test_null_rating_allowed(self, client, isolated_db, monkeypatch, tmp_path):
        """User closed the popup without rating — rating should be NULL."""
        db_path = str(tmp_path / "test_ratings.db")
        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()

        res = client.post("/rating", json={**self._base, "rating": None})
        assert res.status_code == 200
        rows = _query_ratings(db_path)
        assert rows[0][4] is None

    def test_invalid_rating_value_returns_error(self, client):
        res = client.post("/rating", json={**self._base, "rating": 3})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_empty_prefix_returns_error(self, client):
        res = client.post("/rating", json={**self._base, "prefix": ""})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_empty_suggestion_returns_error(self, client):
        res = client.post("/rating", json={**self._base, "suggestion": ""})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_invalid_suggestion_type_returns_error(self, client):
        res = client.post("/rating", json={**self._base, "suggestion_type": "unknown"})
        assert res.status_code == 200
        assert "error" in res.json()

    def test_comment_to_code_type_accepted(self, client, monkeypatch, tmp_path):
        db_path = str(tmp_path / "test_ratings.db")
        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()

        res = client.post("/rating", json={**self._base, "suggestion_type": "comment-to-code", "rating": 1})
        assert res.status_code == 200
        assert "error" not in res.json()

    def test_prefix_and_suffix_stored_in_db(self, client, monkeypatch, tmp_path):
        db_path = str(tmp_path / "test_ratings.db")
        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()

        client.post("/rating", json={**self._base, "rating": 1})

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT prefix, suffix FROM rating").fetchone()
        conn.close()
        assert row[0] == self._base["prefix"]
        assert row[1] == self._base["suffix"]

    def test_missing_prefix_field_returns_422(self, client):
        payload = {k: v for k, v in self._base.items() if k != "prefix"}
        res = client.post("/rating", json=payload)
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_db_creates_table(self, monkeypatch, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()

        conn = sqlite3.connect(db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        assert ("rating",) in tables

    def test_migration_adds_missing_columns(self, monkeypatch, tmp_path):
        """init_db should add new columns to an old-schema DB without crashing."""
        db_path = str(tmp_path / "old.db")
        # Create old schema with only original columns
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

        monkeypatch.setattr(_main, "DB_PATH", db_path)
        _main.init_db()  # should not raise

        conn = sqlite3.connect(db_path)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(rating)").fetchall()]
        conn.close()
        assert "suffix" in cols
        assert "suggestion_type" in cols
        assert "accepted" in cols
        assert "timestamp" in cols
