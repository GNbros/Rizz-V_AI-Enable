#!/usr/bin/env python3
"""
Sequential Multi-Experiment Runner for RV32IM Assembly LLM Pipeline.

Runs train/train_sft.py for each provided YAML config, catches failures
without stopping the queue, and prints a summary table at the end.

CLI usage:
    # Run all experiments in a directory
    python scripts/run_experiment.py configs/experiments/*.yaml

    # Run specific experiments
    python scripts/run_experiment.py \\
        configs/experiments/codegen_lora.yaml \\
        configs/experiments/starcoder_lora.yaml

    # Dry run (validate configs, don't train)
    python scripts/run_experiment.py --dry-run configs/experiments/*.yaml

Input:
    Positional args : paths to YAML config files
    --dry-run       : validate configs only, skip training

Output:
    stdout          : summary table with status, loss, duration per experiment
    runs/experiment_summary.json : machine-readable results
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train" / "train_sft.py"


def validate_config(config_path: str) -> dict:
    """Load and validate a training config file."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return {"valid": False, "error": str(e), "config": None}

    required = ["experiment_name", "base_model_id"]
    missing = [k for k in required if k not in cfg]
    if missing:
        return {
            "valid": False,
            "error": f"Missing required keys: {missing}",
            "config": cfg,
        }

    return {"valid": True, "error": None, "config": cfg}


def run_single_experiment(config_path: str) -> dict:
    """Run a single training experiment via subprocess."""
    result = {
        "config_path": config_path,
        "status": "UNKNOWN",
        "train_loss": None,
        "duration_seconds": None,
        "error": None,
    }

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--config", config_path],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - start
        result["duration_seconds"] = round(elapsed, 1)

        if proc.returncode == 0:
            result["status"] = "PASSED"
            # Try to extract final loss from run_metadata.json
            validation = validate_config(config_path)
            if validation["valid"]:
                exp_name = validation["config"]["experiment_name"]
                output_dir = validation["config"].get("output_dir", "runs/")
                metadata_path = (
                    PROJECT_ROOT / output_dir / exp_name / "run_metadata.json"
                )
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    result["train_loss"] = meta.get("final_train_loss")
        else:
            result["status"] = "FAILED"
            result["error"] = proc.stderr[-500:] if proc.stderr else "Unknown error"

    except Exception as e:
        elapsed = time.time() - start
        result["duration_seconds"] = round(elapsed, 1)
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def format_duration(seconds: Optional[float]) -> str:
    """Format seconds to human-readable duration."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"


def print_summary(results: list[dict]):
    """Print a formatted summary table."""
    # Column widths
    name_w = max(20, max(len(r.get("experiment_name", "?")) for r in results) + 2)
    header = (
        f"{'Experiment':<{name_w}} {'Status':<10} {'Train Loss':<12} {'Duration':<10}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        name = r.get("experiment_name", "?")
        status = r["status"]
        loss = f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "—"
        dur = format_duration(r["duration_seconds"])
        print(f"{name:<{name_w}} {status:<10} {loss:<12} {dur:<10}")

    print(sep)

    passed = sum(1 for r in results if r["status"] == "PASSED")
    total = len(results)
    print(f"\n{passed}/{total} experiments passed\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py configs/experiments/*.yaml
  python scripts/run_experiment.py --dry-run configs/experiments/*.yaml
  python scripts/run_experiment.py \\
      configs/experiments/codegen_lora.yaml \\
      configs/experiments/starcoder_lora.yaml
        """,
    )
    parser.add_argument(
        "configs", nargs="+", help="Paths to YAML experiment config files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs only, don't run training",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Validate all configs first
    # ------------------------------------------------------------------
    experiments = []
    all_valid = True

    for cfg_path in args.configs:
        v = validate_config(cfg_path)
        exp_name = v["config"]["experiment_name"] if v["config"] else Path(cfg_path).stem
        print(f"  {'✓' if v['valid'] else '✗'} {exp_name} ({cfg_path})")
        if not v["valid"]:
            print(f"    ERROR: {v['error']}")
            all_valid = False
        experiments.append(
            {"config_path": cfg_path, "experiment_name": exp_name, "valid": v["valid"]}
        )

    if not all_valid:
        print("\n⚠ Some configs failed validation. Fix them before running.")
        if not args.dry_run:
            print("  Continuing with valid configs only...\n")

    if args.dry_run:
        valid_count = sum(1 for e in experiments if e["valid"])
        print(f"\nDry run complete: {valid_count}/{len(experiments)} configs valid")
        return 0

    # ------------------------------------------------------------------
    # 2. Run experiments sequentially
    # ------------------------------------------------------------------
    results = []
    valid_experiments = [e for e in experiments if e["valid"]]

    for i, exp in enumerate(valid_experiments, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(valid_experiments)}] Running: {exp['experiment_name']}")
        print(f"  Config: {exp['config_path']}")
        print(f"{'='*60}\n")

        result = run_single_experiment(exp["config_path"])
        result["experiment_name"] = exp["experiment_name"]
        results.append(result)

        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        print(f"\n  {status_icon} {exp['experiment_name']}: {result['status']}")
        if result["error"]:
            print(f"    Error: {result['error'][:200]}")

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    print_summary(results)

    # ------------------------------------------------------------------
    # 4. Save machine-readable summary
    # ------------------------------------------------------------------
    summary_path = PROJECT_ROOT / "runs" / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASSED"),
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    # Return non-zero if any experiment failed
    failed = sum(1 for r in results if r["status"] != "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
