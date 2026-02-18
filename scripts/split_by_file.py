#!/usr/bin/env python3
"""
split_by_file.py – Deterministically split normalized .S files into train/val/test.

Input:  Directory of normalized .S files  (paths.normalized_dir)
Output: Three text files listing filenames  (paths.splits_dir/{train,val,test}_files.txt)

CLI usage:
    python scripts/split_by_file.py --config configs/dataset.yaml
    python scripts/split_by_file.py --config configs/dataset.yaml --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically split normalized .S files into train/val/test.",
        epilog=textwrap.dedent("""\
            examples:
              python scripts/split_by_file.py --config configs/dataset.yaml
              python scripts/split_by_file.py --config configs/dataset.yaml --dry-run
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to dataset.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    norm_dir = project_root / cfg["paths"]["normalized_dir"]
    splits_dir = project_root / cfg["paths"]["splits_dir"]
    seed = cfg.get("seed", 42)
    split_cfg = cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1})

    # Gather all .S files
    if not norm_dir.exists():
        print(f"ERROR: normalized dir not found: {norm_dir}", file=sys.stderr)
        print("       Run normalize_asm.py first.", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(f.name for f in norm_dir.glob("*.S"))
    if not all_files:
        print(f"ERROR: no .S files in {norm_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[split] Found {len(all_files)} normalized files (seed={seed})")

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(all_files)

    # Compute split indices
    n = len(all_files)
    train_end = int(n * split_cfg["train"])
    val_end = train_end + int(n * split_cfg["val"])

    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:],
    }

    # Print summary
    for name, flist in splits.items():
        print(f"  {name:5s}: {len(flist):4d} files  ({len(flist)/n*100:.1f}%)")

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, flist in splits.items():
        out_path = splits_dir / f"{name}_files.txt"
        out_path.write_text("\n".join(flist) + "\n", encoding="utf-8")
        print(f"[split] Wrote {out_path}")

    print("[split] Done.")


if __name__ == "__main__":
    main()
