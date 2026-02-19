#!/usr/bin/env python3
"""
split_by_file.py – Deterministically split normalized .S files into train/val/test.
Respest dedup clusters to ensure near-duplicates stay in same split.

Input:
    - Directory of normalized .S files  (paths.normalized_dir)
    - Dedup report                      (data/dedup_report.json, optional but recommended)
Output: Three text files listing filenames  (paths.splits_dir/{train,val,test}_files.txt)

CLI usage:
    python scripts/split_by_file.py --config configs/dataset.yaml
    python scripts/split_by_file.py --config configs/dataset.yaml --dedup-report data/dedup_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

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
              python scripts/split_by_file.py --config configs/dataset.yaml --dedup-report data/dedup_report.json
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to dataset.yaml")
    parser.add_argument("--dedup-report", help="Path to dedup_report.json (if skip, assumes no dedup)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    norm_dir = project_root / cfg["paths"]["normalized_dir"]
    splits_dir = project_root / cfg["paths"]["splits_dir"]
    seed = cfg.get("split", {}).get("seed", 42)
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

    # Load dedup clusters if available
    file_to_cluster: Dict[str, str] = {}
    if args.dedup_report:
        dedup_path = Path(args.dedup_report)
        if dedup_path.exists():
            with open(dedup_path, "r") as f:
                report = json.load(f)
                file_clusters = report.get("file_clusters", {})
                # Convert cluster ID to string
                for fname, cid in file_clusters.items():
                    file_to_cluster[fname] = str(cid)
            print(f"[split] Loaded dedup report: {len(file_clusters)} clustered files")
        else:
            print(f"WARNING: Dedup report {args.dedup_report} not found. Proceeding without dedup constraints.", file=sys.stderr)

    # Group files by cluster (or self if singleton)
    # clusters maps cluster_id -> list of filenames
    clusters: Dict[str, List[str]] = {}
    
    for fname in all_files:
        # If file is in a dedup cluster, use that ID. Otherwise use filename as unique ID.
        cid = file_to_cluster.get(fname, f"singleton_{fname}")
        clusters.setdefault(cid, []).append(fname)

    cluster_ids = sorted(clusters.keys())
    print(f"[split] Found {len(all_files)} files in {len(cluster_ids)} atomic units (seed={seed})")

    # Deterministic shuffle of CLUSTERS
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)

    # Compute split indices based on *number of clusters* (approximate file count balance)
    n = len(cluster_ids)
    train_end = int(n * split_cfg["train"])
    val_end = train_end + int(n * split_cfg["val"])

    split_clusters = {
        "train": cluster_ids[:train_end],
        "val": cluster_ids[train_end:val_end],
        "test": cluster_ids[val_end:],
    }

    # Flatten back to filenames
    final_splits: Dict[str, List[str]] = {k: [] for k in split_clusters}
    for split_name, cids in split_clusters.items():
        for cid in cids:
            final_splits[split_name].extend(clusters[cid])
        final_splits[split_name].sort()

    # Print summary
    total_assigned = sum(len(x) for x in final_splits.values())
    for name, flist in final_splits.items():
        print(f"  {name:5s}: {len(flist):4d} files  ({len(flist)/total_assigned*100:.1f}%)")

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, flist in final_splits.items():
        out_path = splits_dir / f"{name}_files.txt"
        out_path.write_text("\n".join(flist) + "\n", encoding="utf-8")
        print(f"[split] Wrote {out_path}")

    print("[split] Done.")


if __name__ == "__main__":
    main()
