"""
Unit tests for scripts/eval_text.py
====================================
Tests opcode extraction helpers and opcode set loading.
Full integration tests require a model checkpoint and are skipped by default.

Run:
    pytest tests/test_eval_text.py -v
"""

import pytest
import yaml
from pathlib import Path


# ── Test opcode set from config ──────────────────────────────────────────────

class TestOpcodeConfig:
    """Verify that configs/dataset.yaml contains the expected opcodes."""

    @pytest.fixture
    def config(self):
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml"
        if not cfg_path.exists():
            pytest.skip("configs/dataset.yaml not found")
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    def test_has_rv32im_opcodes(self, config):
        assert "rv32im_opcodes" in config
        opcodes = config["rv32im_opcodes"]
        assert isinstance(opcodes, list)
        assert len(opcodes) > 30  # should have at least the RV32I base

    def test_core_opcodes_present(self, config):
        opcodes = {op.lower() for op in config["rv32im_opcodes"]}
        core = {"add", "sub", "lw", "sw", "beq", "jal", "ret", "addi", "mul"}
        assert core.issubset(opcodes), f"Missing core opcodes: {core - opcodes}"

    def test_no_duplicates(self, config):
        opcodes = config["rv32im_opcodes"]
        assert len(opcodes) == len(set(opcodes)), "Duplicate opcodes in config"


# ── Test opcode accuracy helpers ─────────────────────────────────────────────

class TestOpcodeHelpers:
    """Test the get_opcodes helper from eval_text."""

    def test_get_opcodes_from_config(self):
        from pipeline.scripts.eval_text import get_opcodes
        config = {"rv32im_opcodes": ["add", "sub", "lw"]}
        result = get_opcodes(config)
        assert result == {"add", "sub", "lw"}

    def test_get_opcodes_fallback(self):
        from pipeline.scripts.eval_text import get_opcodes
        result = get_opcodes({})  # no rv32im_opcodes key
        assert "add" in result
        assert "ret" in result
        assert len(result) > 30

    def test_load_config_missing_file(self):
        from pipeline.scripts.eval_text import load_config
        config = load_config("/nonexistent/path.yaml")
        assert isinstance(config, dict)  # should return empty dict or from default


# ── Test JSONL loading ───────────────────────────────────────────────────────

class TestJsonlLoading:
    def test_load_jsonl(self, tmp_path):
        from pipeline.scripts.eval_text import load_jsonl
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "add a0, a1, a2"}\n{"text": "ret"}\n')
        records = load_jsonl(str(f))
        assert len(records) == 2
        assert records[0]["text"] == "add a0, a1, a2"

    def test_load_jsonl_empty_lines(self, tmp_path):
        from pipeline.scripts.eval_text import load_jsonl
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "nop"}\n\n{"text": "ret"}\n\n')
        records = load_jsonl(str(f))
        assert len(records) == 2


# ── Test eval_decoding config ────────────────────────────────────────────────

class TestEvalDecodingConfig:
    """Verify eval_decoding defaults are in configs/dataset.yaml."""

    def test_eval_decoding_present(self):
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml"
        if not cfg_path.exists():
            pytest.skip("configs/dataset.yaml not found")
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        assert "eval_decoding" in config
        dec = config["eval_decoding"]
        assert "seed" in dec
        assert "temperature" in dec
        assert "top_p" in dec
        assert "max_new_tokens" in dec
