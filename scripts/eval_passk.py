#!/usr/bin/env python3
"""
Pass@k Evaluation for RV32IM Assembly LLM
==========================================
Generates k candidates per test sample using seeded temperature sampling,
tests each candidate by assembling and (optionally) ABI-linting, and
computes the unbiased pass@k estimator.

Input / Output Schema
---------------------
Input:
    --checkpoint  : path to HF model checkpoint
    --test-file   : path to test JSONL
    --k           : comma-separated k values (default: "1,5")
    --gcc         : path to riscv gcc binary
    --abi-lint    : flag to enable ABI lint gating
    --output-dir  : directory for metrics.json

Output (fields merged into metrics.json):
    {
        "pass_at_1":           float,
        "pass_at_5":           float,
        "pass_at_1_abi":       float | null,
        "pass_at_5_abi":       float | null,
        "passk_total_samples": int,
        "passk_k_values":      [int],
        "passk_seed":          int,
        "passk_decoding": {
            "temperature": float,
            "top_p": float,
            "max_new_tokens": int
        },
        "passk_timestamp":     str
    }

CLI usage examples
------------------
    python -m scripts.eval_passk \\
        --checkpoint runs/exp01/checkpoint-best \\
        --test-file data/processed/dataset/test.jsonl \\
        --k 1,5 \\
        --gcc riscv64-unknown-elf-gcc \\
        --output-dir runs/exp01/ \\
        --seed 42

    # With ABI lint gating:
    python -m scripts.eval_passk \\
        --checkpoint runs/exp01/checkpoint-best \\
        --test-file data/processed/dataset/test.jsonl \\
        --k 1,5 --abi-lint \\
        --gcc riscv64-unknown-elf-gcc \\
        --output-dir runs/exp01/
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import sibling modules
from scripts.abi_lint import lint_asm
from scripts.eval_assemble import (
    REBUILD_FNS,
    extract_prompt,
    generate_text,
    load_config,
    load_jsonl,
    rebuild_asm_causal,
    try_assemble,
)


# ── Pass@k estimator ────────────────────────────────────────────────────────

def _comb(n: int, k: int) -> float:
    """Compute C(n, k) using math.comb, with edge-case handling."""
    if n < 0 or k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator.

    Parameters
    ----------
    n : int  – total candidates generated
    c : int  – number of candidates that pass
    k : int  – k value

    Returns
    -------
    float – estimated pass@k in [0, 1]
    """
    if n < k:
        # Not enough candidates; fall back to simple ratio
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


# ── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_passk(
    checkpoint: str,
    test_file: str,
    k_values: list[int],
    gcc_path: str,
    output_dir: str,
    use_abi_lint: bool = False,
    config_path: str | None = None,
    seed: int = 42,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """
    For each test sample, generate max(k_values) candidates,
    test each via assembly + optional ABI lint, compute pass@k.
    """
    config = load_config(config_path)
    max_k = max(k_values)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    records = load_jsonl(test_file)
    total_samples = len(records)

    # Per-sample results
    per_sample_asm_pass: list[int] = []     # count of asm-passing candidates
    per_sample_abi_pass: list[int] = []     # count of asm+abi-passing candidates

    for sample_idx, rec in enumerate(records):
        prompt = extract_prompt(rec, config)
        mode = rec.get("meta", {}).get("mode", "causal")
        rebuild_fn = REBUILD_FNS.get(mode, rebuild_asm_causal)

        asm_passes = 0
        abi_passes = 0

        for candidate_idx in range(max_k):
            # Deterministic per-candidate seed
            candidate_seed = seed + sample_idx * max_k + candidate_idx

            generated = generate_text(
                model, tokenizer, prompt,
                seed=candidate_seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            asm_text = rebuild_fn(rec, generated, config)
            success, stderr, category = try_assemble(asm_text, gcc_path)

            if success:
                asm_passes += 1

                if use_abi_lint:
                    lint_result = lint_asm(asm_text)
                    if lint_result.passed:
                        abi_passes += 1
                else:
                    abi_passes += 1  # no gating = auto-pass

        per_sample_asm_pass.append(asm_passes)
        per_sample_abi_pass.append(abi_passes)

    # Compute pass@k for each k
    results: dict[str, Any] = {}

    for k in k_values:
        # Assembly-only pass@k
        scores = [pass_at_k(max_k, c, k) for c in per_sample_asm_pass]
        results[f"pass_at_{k}"] = round(sum(scores) / max(len(scores), 1), 6)

        if use_abi_lint:
            abi_scores = [pass_at_k(max_k, c, k) for c in per_sample_abi_pass]
            results[f"pass_at_{k}_abi"] = round(sum(abi_scores) / max(len(abi_scores), 1), 6)
        else:
            results[f"pass_at_{k}_abi"] = None

    results["passk_total_samples"] = total_samples
    results["passk_k_values"] = k_values
    results["passk_seed"] = seed
    results["passk_decoding"] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    results["passk_timestamp"] = datetime.now(timezone.utc).isoformat()

    # Merge into metrics.json
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"

    existing: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            existing = json.load(f)

    existing.update(results)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval_passk",
        description="Pass@k evaluation for RV32IM assembly models.",
        epilog="""
CLI usage examples:
  python -m scripts.eval_passk \\
      --checkpoint runs/exp01/checkpoint-best \\
      --test-file data/processed/dataset/test.jsonl \\
      --k 1,5 --gcc riscv64-unknown-elf-gcc \\
      --output-dir runs/exp01/ --seed 42

  # With ABI lint gating:
  python -m scripts.eval_passk \\
      --checkpoint runs/exp01/checkpoint-best \\
      --test-file data/processed/dataset/test.jsonl \\
      --k 1,5 --abi-lint \\
      --gcc riscv64-unknown-elf-gcc \\
      --output-dir runs/exp01/
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="HF model checkpoint path or name")
    parser.add_argument("--test-file", required=True, help="Path to test.jsonl")
    parser.add_argument("--k", default="1,5", help="Comma-separated k values (default: 1,5)")
    parser.add_argument("--abi-lint", action="store_true", help="Gate pass by ABI lint in addition to assembly")
    parser.add_argument("--gcc", default="riscv64-unknown-elf-gcc", help="Path to RISC-V gcc")
    parser.add_argument("--output-dir", required=True, help="Output directory for metrics.json")
    parser.add_argument("--config", default=None, help="Path to configs/dataset.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling (default: 0.95)")

    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k.split(",")]

    print(f"[eval_passk] checkpoint={args.checkpoint}")
    print(f"[eval_passk] k={k_values}  abi_lint={args.abi_lint}")
    print(f"[eval_passk] seed={args.seed}  temp={args.temperature}  top_p={args.top_p}")

    if not shutil.which(args.gcc):
        print(f"WARNING: {args.gcc} not found in PATH.", file=sys.stderr)

    results = evaluate_passk(
        checkpoint=args.checkpoint,
        test_file=args.test_file,
        k_values=k_values,
        gcc_path=args.gcc,
        output_dir=args.output_dir,
        use_abi_lint=args.abi_lint,
        config_path=args.config,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"\n[eval_passk] Results:")
    for k in k_values:
        print(f"  pass@{k} = {results[f'pass_at_{k}']:.4f}", end="")
        if results.get(f"pass_at_{k}_abi") is not None:
            print(f"  (abi-gated: {results[f'pass_at_{k}_abi']:.4f})", end="")
        print()
    print(f"  → metrics.json written to {args.output_dir}")


if __name__ == "__main__":
    main()
