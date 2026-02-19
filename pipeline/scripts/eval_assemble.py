#!/usr/bin/env python3
"""
Assemble-in-Context Evaluation for RV32IM Assembly LLM
======================================================
For each test sample, reconstructs a full .S file (mode-dependent),
compiles with riscv64-unknown-elf-gcc, and categorises failures.

Input / Output Schema
---------------------
Input:
    --checkpoint  : path to HF model checkpoint
    --test-file   : path to test JSONL
    --gcc         : path to riscv gcc binary
    --output-dir  : directory for metrics.json + error_histogram.json
    --config      : path to configs/dataset.yaml

Output (metrics.json fields appended):
    "assemble_total":      int,
    "assemble_pass":       int,
    "assemble_pass_rate":  float

Output (error_histogram.json):
    {
        "seed": int,
        "decoding": {...},
        "histogram": {"unknown_opcode": int, "syntax_error": int, ...},
        "top_errors": [{"stderr": str, "count": int}, ...],
        "timestamp": str
    }

CLI usage examples
------------------
    python -m scripts.eval_assemble \\
        --checkpoint runs/exp01/checkpoint-best \\
        --test-file data/processed/dataset/test.jsonl \\
        --gcc riscv64-unknown-elf-gcc \\
        --output-dir runs/exp01/ \\
        --seed 42

    # With pre-generated predictions (skip model inference):
    python -m scripts.eval_assemble \\
        --predictions runs/exp01/predictions.jsonl \\
        --gcc riscv64-unknown-elf-gcc \\
        --output-dir runs/exp01/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Config loader (shared with eval_text) ────────────────────────────────────

def load_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        default = Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml"
        if default.exists():
            config_path = str(default)
        else:
            return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Scaffold builders (mode-dependent .S reconstruction) ─────────────────────

def rebuild_asm_fim(record: dict, generated: str, config: dict) -> str:
    """FIM mode: insert generated text between prefix and suffix."""
    tokens = config.get("special_tokens", {})
    prefix_tok = tokens.get("fim_prefix", "<fim_prefix>")
    suffix_tok = tokens.get("fim_suffix", "<fim_suffix>")
    middle_tok = tokens.get("fim_middle", "<fim_middle>")

    text = record["text"]
    # Try to extract real prefix/suffix from the training format
    if prefix_tok in text and suffix_tok in text:
        parts = text.split(prefix_tok, 1)
        after_prefix = parts[1] if len(parts) > 1 else ""
        if suffix_tok in after_prefix:
            prefix = after_prefix.split(suffix_tok, 1)[0]
            suffix_part = after_prefix.split(suffix_tok, 1)[1]
            suffix = suffix_part.split(middle_tok, 1)[0] if middle_tok in suffix_part else suffix_part
            return prefix + generated + suffix

    # Fallback: just use generated text
    return generated


def rebuild_asm_block(record: dict, generated: str, config: dict) -> str:
    """Block mode: prefix + generated continuation."""
    tokens = config.get("special_tokens", {})
    block_prefix = tokens.get("block_prefix", "<block_prefix>")
    block_end = tokens.get("block_end", "<block_end>")

    text = record["text"]
    if block_prefix in text:
        prefix = text.split(block_prefix, 1)[1].split(block_end, 1)[0]
        return prefix + "\n" + generated
    return generated


def rebuild_asm_function(record: dict, generated: str, config: dict) -> str:
    """Function mode: wrap generated body in function scaffold."""
    meta = record.get("meta", {})
    func_name = meta.get("function", "generated_func")
    return f".globl {func_name}\n{func_name}:\n{generated}\n"


def rebuild_asm_causal(record: dict, generated: str, config: dict) -> str:
    """Causal mode: generated text is the entire assembly."""
    return generated


REBUILD_FNS = {
    "fim": rebuild_asm_fim,
    "block": rebuild_asm_block,
    "function": rebuild_asm_function,
    "causal": rebuild_asm_causal,
}


# ── GCC assembly ─────────────────────────────────────────────────────────────

ERROR_PATTERNS = [
    ("unknown_opcode", re.compile(r"(unrecognized opcode|illegal operand|unknown mnemonic)", re.I)),
    ("syntax_error",   re.compile(r"(syntax error|junk at end|expected|parse error)", re.I)),
    ("undefined_symbol", re.compile(r"(undefined reference|undeclared|undefined symbol)", re.I)),
    ("relocation_error", re.compile(r"relocation", re.I)),
]


def categorise_error(stderr: str) -> str:
    """Map gcc stderr to an error category."""
    for category, pattern in ERROR_PATTERNS:
        if pattern.search(stderr):
            return category
    return "other"


def try_assemble(
    asm_text: str,
    gcc_path: str = "riscv64-unknown-elf-gcc",
    timeout: int = 30,
) -> tuple[bool, str, str]:
    """
    Write asm_text to a temp .S file, try to assemble with gcc.

    Returns (success: bool, stderr: str, category: str).
    """
    with tempfile.NamedTemporaryFile(suffix=".S", mode="w", delete=False) as f:
        f.write(asm_text)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [gcc_path, "-c", "-march=rv32im", "-mabi=ilp32", "-o", "/dev/null", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        stderr = result.stderr.strip()
        category = "" if success else categorise_error(stderr)
        return success, stderr, category
    except FileNotFoundError:
        return False, f"gcc not found: {gcc_path}", "toolchain_missing"
    except subprocess.TimeoutExpired:
        return False, "gcc timed out", "timeout"
    finally:
        os.unlink(tmp_path)


# ── Model generation ─────────────────────────────────────────────────────────

def generate_text(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: torch.device | None = None,
) -> str:
    """Generate text from a prompt with seeded decoding."""
    torch.manual_seed(seed)
    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def extract_prompt(record: dict, config: dict) -> str:
    """Extract the prompt portion from a JSONL record based on mode."""
    mode = record.get("meta", {}).get("mode", "causal")
    text = record["text"]
    tokens = config.get("special_tokens", {})

    if mode == "fim":
        middle_tok = tokens.get("fim_middle", "<fim_middle>")
        if middle_tok in text:
            return text.split(middle_tok)[0] + middle_tok
    elif mode == "block":
        block_prefix = tokens.get("block_prefix", "<block_prefix>")
        block_end = tokens.get("block_end", "<block_end>")
        if block_prefix in text:
            return text.split(block_end)[0] if block_end in text else text
    elif mode == "function":
        asm_open = tokens.get("asm_open", "<asm>")
        if asm_open in text:
            return text.split(asm_open)[0] + asm_open

    # Causal / fallback: use first 50% of text as prompt
    lines = text.split("\n")
    mid = max(1, len(lines) // 2)
    return "\n".join(lines[:mid])


# ── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_assemble(
    checkpoint: str | None,
    test_file: str | None,
    predictions_file: str | None,
    gcc_path: str,
    output_dir: str,
    config_path: str | None = None,
    seed: int = 42,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Run assemble-in-context evaluation."""

    config = load_config(config_path)
    decoding_cfg = config.get("eval_decoding", {})
    # CLI args override config defaults
    seed = seed or decoding_cfg.get("seed", 42)
    max_new_tokens = max_new_tokens or decoding_cfg.get("max_new_tokens", 256)
    temperature = temperature or decoding_cfg.get("temperature", 0.8)
    top_p = top_p or decoding_cfg.get("top_p", 0.95)

    # Load predictions or generate them
    if predictions_file:
        predictions = load_jsonl(predictions_file)
    else:
        assert checkpoint and test_file, "Need --checkpoint and --test-file if no --predictions"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        records = load_jsonl(test_file)
        predictions = []
        for idx, rec in enumerate(records):
            prompt = extract_prompt(rec, config)
            generated = generate_text(
                model, tokenizer, prompt, seed=seed + idx,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            )
            mode = rec.get("meta", {}).get("mode", "causal")
            rebuild_fn = REBUILD_FNS.get(mode, rebuild_asm_causal)
            asm_text = rebuild_fn(rec, generated, config)
            predictions.append({
                "asm": asm_text,
                "mode": mode,
                "meta": rec.get("meta", {}),
                "generated": generated,
            })

    # Assemble each prediction
    total = len(predictions)
    passed = 0
    error_cats: Counter = Counter()
    raw_errors: Counter = Counter()

    for pred in predictions:
        asm = pred.get("asm", pred.get("text", ""))
        success, stderr, category = try_assemble(asm, gcc_path)
        if success:
            passed += 1
        else:
            error_cats[category] += 1
            # Truncate raw errors for histogram
            short_err = stderr[:200] if stderr else "(empty)"
            raw_errors[short_err] += 1

    pass_rate = passed / max(total, 1)

    # ── Write metrics ────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    existing: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    existing.update({
        "assemble_total": total,
        "assemble_pass": passed,
        "assemble_pass_rate": round(pass_rate, 6),
        "assemble_seed": seed,
        "assemble_decoding": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "assemble_timestamp": datetime.now(timezone.utc).isoformat(),
    })

    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    # Error histogram
    top_errors = [{"stderr": k, "count": v} for k, v in raw_errors.most_common(10)]
    histogram = {
        "seed": seed,
        "decoding": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "histogram": dict(error_cats),
        "top_errors": top_errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    hist_path = out_dir / "error_histogram.json"
    with open(hist_path, "w") as f:
        json.dump(histogram, f, indent=2)

    return existing


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval_assemble",
        description="Assemble-in-context evaluation for RV32IM assembly models.",
        epilog="""
CLI usage examples:
  python -m scripts.eval_assemble \\
      --checkpoint runs/exp01/checkpoint-best \\
      --test-file data/processed/dataset/test.jsonl \\
      --gcc riscv64-unknown-elf-gcc \\
      --output-dir runs/exp01/ --seed 42

  # With pre-generated predictions:
  python -m scripts.eval_assemble \\
      --predictions runs/exp01/predictions.jsonl \\
      --gcc riscv64-unknown-elf-gcc \\
      --output-dir runs/exp01/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None, help="HF model checkpoint path or name")
    parser.add_argument("--test-file", default=None, help="Path to test.jsonl")
    parser.add_argument("--predictions", default=None, help="Pre-generated predictions JSONL (skip inference)")
    parser.add_argument("--gcc", default="riscv64-unknown-elf-gcc", help="Path to RISC-V gcc (default: riscv64-unknown-elf-gcc)")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics.json + error_histogram.json")
    parser.add_argument("--config", default=None, help="Path to configs/dataset.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for nucleus sampling (default: 0.95)")

    args = parser.parse_args()

    if not args.predictions and (not args.checkpoint or not args.test_file):
        parser.error("Either --predictions or both --checkpoint and --test-file required")

    print(f"[eval_assemble] gcc={args.gcc}")
    print(f"[eval_assemble] seed={args.seed}  temp={args.temperature}  top_p={args.top_p}")

    # Verify gcc exists
    if not shutil.which(args.gcc):
        print(f"WARNING: {args.gcc} not found in PATH. Assembly checks will fail.", file=sys.stderr)

    metrics = evaluate_assemble(
        checkpoint=args.checkpoint,
        test_file=args.test_file,
        predictions_file=args.predictions,
        gcc_path=args.gcc,
        output_dir=args.output_dir,
        config_path=args.config,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"\n[eval_assemble] Results:")
    print(f"  pass_rate={metrics['assemble_pass_rate']:.4f} ({metrics['assemble_pass']}/{metrics['assemble_total']})")
    print(f"  → metrics.json + error_histogram.json written to {args.output_dir}")


if __name__ == "__main__":
    main()
