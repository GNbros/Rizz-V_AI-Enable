#!/usr/bin/env python3
"""
build_dataset.py – Build JSONL datasets in four training modes from normalized .S files.

Input:
    - Normalized .S files         (paths.normalized_dir)
    - Split file lists            (paths.splits_dir/{train,val,test}_files.txt)
    - Config                      (configs/dataset.yaml)

Output:
    - data/processed/dataset/train.jsonl
    - data/processed/dataset/val.jsonl
    - data/processed/dataset/test.jsonl
    - data/processed/dataset/manifest.json

JSONL record schema (one JSON object per line):
    {
        "text": "<string>",
        "meta": {
            "file":       "<string>",
            "function":   "<string | null>",
            "span_lines": [<int>, <int>],
            "mode":       "causal | fim | block | function"
        }
    }

CLI usage:
    python scripts/build_dataset.py --config configs/dataset.yaml
    python scripts/build_dataset.py --config configs/dataset.yaml --dry-run
    python scripts/build_dataset.py --config configs/dataset.yaml --splits train,val
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Sample generators ────────────────────────────────────────────────────────

def make_causal_sample(
    text: str,
    filename: str,
    span: Tuple[int, int],
    func_name: Optional[str],
) -> Dict[str, Any]:
    """Causal LM: raw text, no special tokens."""
    return {
        "text": text,
        "meta": {
            "file": filename,
            "function": func_name,
            "span_lines": list(span),
            "mode": "causal",
        },
    }


def make_fim_sample(
    text: str,
    filename: str,
    span: Tuple[int, int],
    func_name: Optional[str],
    tokens: dict,
    rng,
    min_prefix: int = 1,
    min_suffix: int = 1,
) -> Optional[Dict[str, Any]]:
    """Fill-in-the-middle: <fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"""
    lines = text.splitlines(keepends=True)
    if len(lines) < min_prefix + min_suffix + 1:
        return None  # too short for FIM

    # Random split point
    split_start = rng.randint(min_prefix, len(lines) - min_suffix - 1)
    split_end = rng.randint(split_start + 1, len(lines) - min_suffix)

    prefix = "".join(lines[:split_start])
    middle = "".join(lines[split_start:split_end])
    suffix = "".join(lines[split_end:])

    combined = (
        tokens["fim_prefix"] + prefix
        + tokens["fim_suffix"] + suffix
        + tokens["fim_middle"] + middle
    )

    return {
        "text": combined,
        "meta": {
            "file": filename,
            "function": func_name,
            "span_lines": list(span),
            "mode": "fim",
        },
    }


def make_block_sample(
    text: str,
    filename: str,
    span: Tuple[int, int],
    func_name: Optional[str],
    tokens: dict,
    rng,
    min_prefix_lines: int = 2,
    max_prefix_ratio: float = 0.8,
) -> Optional[Dict[str, Any]]:
    """Block completion: <block_prefix>{prefix}<block_end>"""
    lines = text.splitlines(keepends=True)
    max_prefix = max(min_prefix_lines, int(len(lines) * max_prefix_ratio))
    if len(lines) < min_prefix_lines + 1:
        return None

    cut = rng.randint(min_prefix_lines, min(max_prefix, len(lines) - 1))
    prefix = "".join(lines[:cut])

    combined = tokens["block_prefix"] + prefix + tokens["block_end"]

    return {
        "text": combined,
        "meta": {
            "file": filename,
            "function": func_name,
            "span_lines": list(span),
            "mode": "block",
        },
    }


def extract_functions(
    text: str,
    label_pattern: str,
    include_directives: bool,
) -> List[Tuple[str, str, Tuple[int, int]]]:
    """Extract function-like blocks from assembly text.

    Returns list of (label_name, body_text, (start_line, end_line)).
    """
    label_re = re.compile(label_pattern)
    lines = text.splitlines(keepends=True)
    functions: List[Tuple[str, str, Tuple[int, int]]] = []
    current_label: Optional[str] = None
    current_start: int = 0
    current_body: List[str] = []

    # Directive lines we may include before a label as part of its "signature"
    directive_re = re.compile(r"^\s*\.(globl|global|type|size|align)\b")

    pending_directives: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Collect directives that might belong to the next function
        if include_directives and directive_re.match(stripped):
            pending_directives.append(line)
            continue

        m = label_re.match(stripped)
        if m:
            # Save previous function
            if current_label and current_body:
                body = "".join(current_body)
                if body.strip():
                    functions.append((current_label, body, (current_start, i - 1)))

            current_label = stripped.rstrip(":")
            current_start = i - len(pending_directives)
            current_body = list(pending_directives)  # include directives
            pending_directives = []
            current_body.append(line)
        else:
            pending_directives = []  # discard unrelated directives
            if current_label:
                current_body.append(line)

    # Flush last function
    if current_label and current_body:
        body = "".join(current_body)
        if body.strip():
            functions.append((current_label, body, (current_start, len(lines) - 1)))

    return functions


def make_function_sample(
    func_name: str,
    func_body: str,
    filename: str,
    span: Tuple[int, int],
    tokens: dict,
) -> Dict[str, Any]:
    """Function-level: <task>{signature}</task><asm>{body}<eof_func>"""
    # Use the label line as the "signature/task description"
    body_lines = func_body.splitlines(keepends=True)
    if not body_lines:
        signature = func_name
        body = ""
    else:
        signature = body_lines[0].strip()
        body = "".join(body_lines[1:]) if len(body_lines) > 1 else ""

    combined = (
        tokens["task_open"] + signature + tokens["task_close"]
        + tokens["asm_open"] + body + tokens["eof_func"]
    )

    return {
        "text": combined,
        "meta": {
            "file": filename,
            "function": func_name,
            "span_lines": list(span),
            "mode": "function",
        },
    }


# ── Builder core ─────────────────────────────────────────────────────────────

def build_split(
    split_name: str,
    file_list: List[str],
    norm_dir: Path,
    cfg: dict,
    global_seed: int,
) -> List[Dict[str, Any]]:
    """Build all JSONL samples for one split."""
    import random as _random

    tokens = cfg["special_tokens"]
    modes = cfg.get("modes", {"causal": 1.0})
    fim_cfg = cfg.get("fim", {})
    block_cfg = cfg.get("block", {})
    func_cfg = cfg.get("function", {})

    # Build cumulative distribution for mode selection
    mode_names = list(modes.keys())
    mode_weights = [modes[m] for m in mode_names]

    # Per-split seed
    split_seed = global_seed + hash(split_name) % (2**31)
    rng = _random.Random(split_seed)

    samples: List[Dict[str, Any]] = []

    for fname in sorted(file_list):
        fpath = norm_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping", file=sys.stderr)
            continue

        text = fpath.read_text(encoding="utf-8")
        if not text.strip():
            continue

        total_lines = len(text.splitlines())

        # Select mode for this snippet
        mode = rng.choices(mode_names, weights=mode_weights, k=1)[0]

        if mode == "causal":
            sample = make_causal_sample(text, fname, (1, total_lines), None)
            samples.append(sample)

        elif mode == "fim":
            sample = make_fim_sample(
                text, fname, (1, total_lines), None, tokens, rng,
                min_prefix=fim_cfg.get("min_prefix_lines", 1),
                min_suffix=fim_cfg.get("min_suffix_lines", 1),
            )
            if sample:
                samples.append(sample)
            else:
                # Fallback to causal if too short for FIM
                samples.append(make_causal_sample(text, fname, (1, total_lines), None))

        elif mode == "block":
            sample = make_block_sample(
                text, fname, (1, total_lines), None, tokens, rng,
                min_prefix_lines=block_cfg.get("min_prefix_lines", 2),
                max_prefix_ratio=block_cfg.get("max_prefix_ratio", 0.8),
            )
            if sample:
                samples.append(sample)
            else:
                samples.append(make_causal_sample(text, fname, (1, total_lines), None))

        elif mode == "function":
            funcs = extract_functions(
                text,
                func_cfg.get("label_pattern", r"^[a-zA-Z_][a-zA-Z0-9_]*:"),
                func_cfg.get("include_directives", True),
            )
            if funcs:
                for func_name, func_body, span in funcs:
                    sample = make_function_sample(func_name, func_body, fname, span, tokens)
                    samples.append(sample)
            else:
                # No functions found → fallback to causal
                samples.append(make_causal_sample(text, fname, (1, total_lines), None))

    return samples


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build JSONL datasets in four training modes.",
        epilog=textwrap.dedent("""\
            examples:
              python scripts/build_dataset.py --config configs/dataset.yaml
              python scripts/build_dataset.py --config configs/dataset.yaml --dry-run
              python scripts/build_dataset.py --config configs/dataset.yaml --splits train,val
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to dataset.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    parser.add_argument("--splits", default="train,val,test",
                        help="Comma-separated splits to build (default: train,val,test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    norm_dir = project_root / cfg["paths"]["normalized_dir"]
    splits_dir = project_root / cfg["paths"]["splits_dir"]
    dataset_dir = project_root / cfg["paths"]["dataset_dir"]
    global_seed = cfg.get("seed", 42)
    requested_splits = [s.strip() for s in args.splits.split(",")]

    # Validate prerequisites
    for req_dir, label in [(norm_dir, "normalized"), (splits_dir, "splits")]:
        if not req_dir.exists():
            print(f"ERROR: {label} dir not found: {req_dir}", file=sys.stderr)
            print(f"       Run previous pipeline steps first.", file=sys.stderr)
            sys.exit(1)

    if not args.dry_run:
        dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "seed": global_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_tokens_version": cfg.get("special_tokens", {}).get("version", 0),
        "splits": {},
    }

    print(f"[build] seed={global_seed}  modes={cfg.get('modes', {})}")
    print(f"[build] tokens version={cfg.get('special_tokens', {}).get('version', 0)}")

    for split_name in requested_splits:
        list_path = splits_dir / f"{split_name}_files.txt"
        if not list_path.exists():
            print(f"  WARNING: {list_path} not found, skipping {split_name}", file=sys.stderr)
            continue

        file_list = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"\n[build] {split_name}: {len(file_list)} source files")

        samples = build_split(split_name, file_list, norm_dir, cfg, global_seed)

        # Count by mode
        mode_counts: Dict[str, int] = {}
        for s in samples:
            m = s["meta"]["mode"]
            mode_counts[m] = mode_counts.get(m, 0) + 1

        manifest["splits"][split_name] = {"total": len(samples), **mode_counts}

        print(f"  total samples: {len(samples)}")
        for mode_name, count in sorted(mode_counts.items()):
            print(f"    {mode_name:10s}: {count:5d}")

        if not args.dry_run:
            out_path = dataset_dir / f"{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"  → {out_path}")

    # Write manifest
    if not args.dry_run:
        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n[build] Manifest → {manifest_path}")
    else:
        print("\n[dry-run] No files written.")
        print(json.dumps(manifest, indent=2))

    print("[build] Done.")


if __name__ == "__main__":
    main()
