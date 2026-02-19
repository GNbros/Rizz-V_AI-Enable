import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path so we can import main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock transformers entirely before importing main
# This prevents any transformers code from running
sys.modules["transformers"] = MagicMock()
sys.modules["transformers"].pipeline = MagicMock()

# Configure the pipeline mock to return a callable that returns our result
def mock_pipe_func(*args, **kwargs):
    return [{"generated_text": "Mocked assembly code"}]

sys.modules["transformers"].pipeline.return_value = mock_pipe_func

# Now import app (renamed from main)
from app import app, init_db

# Create test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    """Ensure clean DB state for tests"""
    init_db()
    # In a real scenario, use a temporary file or :memory:
    # main.py hardcodes ratings.db, so we test with that for now
    # Ideally refactor main.py to accept db_path env var
    yield

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "RISC-V CodeGen API is running"}

def test_generate_code_mocked():
    payload = {"prompt": "# add 10 to a0", "max_new_tokens": 20}
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200
    assert "generated_code" in response.json()
    assert response.json()["generated_code"] == "Mocked assembly code"

def test_generate_validation_empty_prompt():
    response = client.post("/generate", json={"prompt": ""})
    # FastAPI/Pydantic validation might handle this or custom logic
    # main.py explicitly checks if not req.prompt
    assert response.json() == {"error": "Prompt is required"}

def test_rating_submission():
    payload = {
        "prompt": "# test",
        "suggestion": "li a0, 1",
        "rating": 5
    }
    response = client.post("/rating", json=payload)
    assert response.status_code == 200
    assert response.json()["message"] == "Rating saved"
    assert response.json()["rating"] == 5

def test_rating_validation_bounds():
    payload = {
        "prompt": "# test",
        "suggestion": "li a0, 1",
        "rating": 6  # Invalid
    }
    response = client.post("/rating", json=payload)
    assert response.json() == {"error": "Rating must be between 1 and 5"}
