#!/usr/bin/env python3
"""
Compare Runs Tool
=================
Scans a directory of model runs (output of train_sft.py) and produces a comparison table.

CLI usage:
    python pipeline/scripts/compare_runs.py --runs-dir outputs/runs
    python pipeline/scripts/compare_runs.py --runs-dir runs/ --output comparison.csv
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

def load_run_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {meta_path}: {e}", file=sys.stderr)
        return None

def flatten_metadata(meta: dict) -> dict:
    """Extract key metrics for comparison."""
    # Handle missing keys gracefully
    return {
        "experiment_name": meta.get("experiment_name", "N/A"),
        "base_model": meta.get("base_model_id", "N/A"),
        "training_mode": meta.get("training_mode", "N/A"),
        "status": meta.get("status", "unknown"),
        "train_loss": meta.get("final_train_loss", "N/A"),
        "eval_loss": meta.get("final_eval_loss", "N/A"),
        "duration_s": meta.get("elapsed_seconds", "N/A"),
        "gpu": meta.get("hardware", {}).get("gpu", "N/A"),
        "completed_at": meta.get("completed_at", "N/A"),
        "run_id": meta.get("experiment_name", Path(meta.get("output_dir", ".")).name) # simplified
    }

def main():
    parser = argparse.ArgumentParser(description="Compare model training runs.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subdirectories")
    parser.add_argument("--output", help="Path to save CSV output (optional)")
    args = parser.parse_args()

    runs_path = Path(args.runs_dir)
    if not runs_path.exists():
        print(f"Error: Directory {runs_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {runs_path} for run_metadata.json...")
    
    rows = []
    for run_dir in runs_path.iterdir():
        if run_dir.is_dir():
            meta = load_run_metadata(run_dir)
            if meta:
                rows.append(flatten_metadata(meta))

    if not rows:
        print("No valid runs found.")
        sys.exit(0)

    # Sort by completion time (descending)
    rows.sort(key=lambda x: x.get("completed_at") or "", reverse=True)

    # Define columns
    columns = [
        "experiment_name", "base_model", "training_mode", "status", 
        "train_loss", "eval_loss", "duration_s", "gpu", "completed_at"
    ]

    # Print Table
    # Determine column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col, ""))
            widths[col] = max(widths[col], len(val))

    header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)

    print("\n" + header)
    print(sep)
    for row in rows:
        print(" | ".join(f"{str(row.get(col, '')):<{widths[col]}}" for col in columns))
    print("\n")

    # Save to CSV
    if args.output:
        try:
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                     # Filter row to only include columns
                    filtered = {k: v for k, v in row.items() if k in columns}
                    writer.writerow(filtered)
            print(f"Comparison saved to {args.output}")
        except Exception as e:
            print(f"Error saving CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
