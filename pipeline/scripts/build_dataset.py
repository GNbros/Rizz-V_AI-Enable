#!/usr/bin/env python3
"""
build_dataset.py – Build JSONL datasets in four training modes from normalized .S files.

Input:
    - Raw dataset file            (paths.raw_input)
    - Config                      (configs/dataset.yaml)

Output:
    - data/normalized/*.S         (Individual assembly files)
    - data/splits/{train,val,test}_files.txt
    - data/jsonl/train.jsonl
    - data/jsonl/val.jsonl
    - data/jsonl/test.jsonl
    - data/jsonl/manifest.json

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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import random
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── Normalization & Splitting ────────────────────────────────────────────────

def split_raw_file(raw_path: Path, out_dir: Path) -> List[str]:
    """Splits the large raw text file into individual .S files."""
    if not raw_path.exists():
        print(f"ERROR: Raw file not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[build] Reading raw file: {raw_path}")
    content = raw_path.read_text(encoding="utf-8")
    
    # Split by the header separator found in the file:
    # #*****************************************************************************
    # # fcvt.S
    # #-----------------------------------------------------------------------------
    
    sections = re.split(r'(?m)^#\*{10,}\n#\s+([a-zA-Z0-9_]+\.S)\s*\n', content)
    
    if len(sections) < 3:
        print("WARNING: Could not detect file separators. Treating as single file 'dataset.S'.")
        out_file = out_dir / "dataset.S"
        out_file.write_text(content, encoding="utf-8")
        return ["dataset.S"]

    generated_files = []
    
    for i in range(1, len(sections), 2):
        fname = sections[i].strip()
        body = sections[i+1]
        
        header = f"#*****************************************************************************\n# {fname}\n"
        full_content = header + body
        
        out_path = out_dir / fname
        out_path.write_text(full_content, encoding="utf-8")
        generated_files.append(fname)
        
    print(f"[build] Extracted {len(generated_files)} files to {out_dir}")
    return generated_files

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

# ── Builder core ─────────────────────────────────────────────────────────────

def build_split(
    split_name: str,
    file_list: List[str],
    norm_dir: Path,
    cfg: dict,
    global_seed: int,
) -> List[Dict[str, Any]]:
    """Build all JSONL samples for one split."""
    
    modes = cfg.get("modes", {"causal": 1.0})
    # Filter out 0.0 probability modes
    active_modes = {k: v for k, v in modes.items() if v > 0}
    if not active_modes:
        active_modes = {"causal": 1.0}

    mode_names = list(active_modes.keys())
    mode_weights = [active_modes[m] for m in mode_names]

    # Per-split seed
    split_seed = global_seed + hash(split_name) % (2**31)
    rng = random.Random(split_seed)

    samples: List[Dict[str, Any]] = []

    for fname in sorted(file_list):
        fpath = norm_dir / fname
        if not fpath.exists():
            continue

        text = fpath.read_text(encoding="utf-8")
        if not text.strip():
            continue

        total_lines = len(text.splitlines())

        # Select mode for this snippet
        mode = rng.choices(mode_names, weights=mode_weights, k=1)[0]

        # MVP: Always fallback to causal implementation until other modes are implemented
        sample = make_causal_sample(text, fname, (1, total_lines), None)
        # Record the INTENDED mode in meta, or just 'causal'?
        # For MVP correctness, let's label it 'causal' if we produced causal text.
        # But if the user wants to test the pipeline logic, maybe we should keep the mode label but put causal text?
        # Let's label it 'causal' to avoid confusion.
        sample["meta"]["mode"] = "causal" 
        
        samples.append(sample)

    return samples

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build JSONL datasets in four training modes.",
        epilog=textwrap.dedent("""\
            examples:
              python scripts/build_dataset.py --config configs/dataset.yaml
              python scripts/build_dataset.py --config configs/dataset.yaml --dry-run
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to dataset.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # project_root = Path(args.config).resolve().parent.parent  <-- OLD
    # script is in pipeline/scripts/ => parent.parent.parent = repo root
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Paths - using keys from the rich config
    try:
        raw_file = project_root / cfg["paths"]["raw_input"]
        norm_dir = project_root / cfg["paths"]["normalized_dir"]
        splits_dir = project_root / cfg["paths"]["splits_dir"]
        dataset_dir = project_root / cfg["paths"]["jsonl_dir"]
        
        global_seed = cfg.get("split", {}).get("seed", 42)
        test_ratio = cfg.get("split", {}).get("test", 0.1)
        val_ratio = cfg.get("split", {}).get("val", 0.1)
    except KeyError as e:
        print(f"ERROR: Config missing key {e}. Check dataset.yaml schema.", file=sys.stderr)
        sys.exit(1)

    random.seed(global_seed)

    if not args.dry_run:
        norm_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Split raw file into normalized individual files
    if not args.dry_run:
        all_files = split_raw_file(raw_file, norm_dir)
    else:
        # Dry run simulation
        content = raw_file.read_text(encoding="utf-8") if raw_file.exists() else ""
        sections = re.split(r'(?m)^#\*{10,}\n#\s+([a-zA-Z0-9_]+\.S)\s*\n', content)
        all_files = [sections[i].strip() for i in range(1, len(sections), 2)]
        print(f"[dry-run] Would extract {len(all_files)} files to {norm_dir}")

    # 2. Create Train/Val/Test Lists
    all_files.sort()
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    test_files = all_files[:n_test]
    val_files = all_files[n_test:n_test+n_val]
    train_files = all_files[n_test+n_val:]
    
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    # Write split lists
    if not args.dry_run:
        for name, files in splits.items():
            (splits_dir / f"{name}_files.txt").write_text("\n".join(files), encoding="utf-8")

    manifest = {
        "seed": global_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }

    # 3. Build JSONL
    print(f"\n[build] building JSONL datasets...")
    for split_name, files in splits.items():
        print(f"  {split_name}: {len(files)} source files")
        
        samples = build_split(split_name, files, norm_dir, cfg, global_seed)
        
        manifest["splits"][split_name] = {"total": len(samples)}
        
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

    print("[build] Done.")

if __name__ == "__main__":
    main()
