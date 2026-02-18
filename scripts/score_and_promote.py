#!/usr/bin/env python3
"""
score_and_promote.py — Score model runs and promote the best to HF Hub 'prod'.

Deliverable I/O Schema
=======================
Inputs:
  --metrics-dir  DIR    Directory containing metrics.json files per run
                        (default: ./outputs/metrics)
  --run-tag      TAG    Specific run tag to promote (e.g. run-42)
  --hf-repo      REPO   HF Hub repo id (e.g. GNbros/rizz-v-model)
  --token        TOKEN  HF write token (or set HF_TOKEN env var)
  --auto-best          Auto-select best run by lowest perplexity
  --promote            Actually perform the promotion (otherwise dry-run)
  --seed         INT    Random seed for determinism (default: 42)

Outputs:
  stdout:  JSON summary of selected run + promotion result
  exit 0:  success
  exit 1:  run tag not found
  exit 2:  metrics missing or malformed

CLI Usage Examples
==================
  # Show help
  python scripts/score_and_promote.py --help

  # Dry-run: show best run without promoting
  python scripts/score_and_promote.py \\
      --hf-repo GNbros/rizz-v-model \\
      --auto-best

  # Promote a specific run tag to prod
  python scripts/score_and_promote.py \\
      --hf-repo GNbros/rizz-v-model \\
      --run-tag run-42 \\
      --promote \\
      --token $HF_TOKEN

  # Auto-select best and promote
  python scripts/score_and_promote.py \\
      --hf-repo GNbros/rizz-v-model \\
      --auto-best \\
      --promote
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # optional; only used for seeding

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# HF Hub helpers
# ---------------------------------------------------------------------------

def list_run_revisions(hf_repo: str, token: str | None) -> list[dict]:
    """List all refs matching 'run-*' pattern in the HF repo.

    Returns list of dicts with keys: tag, commit_sha, metrics (if available).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    try:
        refs = api.list_repo_refs(repo_id=hf_repo)
    except Exception as exc:
        print(f"[ERROR] Cannot list refs for {hf_repo}: {exc}", file=sys.stderr)
        return []

    runs = []
    for branch in refs.branches:
        if branch.name.startswith("run-"):
            run_info: dict = {
                "tag": branch.name,
                "commit_sha": branch.target_commit,
                "metrics": None,
            }
            # Try to fetch metrics.json from this revision
            try:
                from huggingface_hub import hf_hub_download

                metrics_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename="metrics.json",
                    revision=branch.name,
                    token=token,
                )
                with open(metrics_path) as f:
                    run_info["metrics"] = json.load(f)
            except Exception:
                pass  # metrics not available for this run
            runs.append(run_info)

    return runs


def select_best_run(runs: list[dict]) -> dict | None:
    """Select run with lowest perplexity from metrics.json.

    Ranking criteria (deterministic, tie-break by tag name):
      1. Lowest perplexity
      2. Highest assemble_pass_rate (if available)
      3. Alphabetically earliest tag
    """
    scored = []
    for run in runs:
        m = run.get("metrics")
        if not m:
            continue
        ppl = m.get("perplexity", float("inf"))
        asm_pass = m.get("assemble_pass_rate", 0.0)
        scored.append((ppl, -asm_pass, run["tag"], run))

    if not scored:
        return None

    scored.sort()  # deterministic: (ppl asc, -asm_pass asc, tag asc)
    return scored[0][3]


def promote_to_prod(hf_repo: str, commit_sha: str, run_tag: str,
                    token: str | None) -> dict:
    """Move the 'prod' branch to point at the given commit.

    Deterministic: deletes existing 'prod' branch, then creates new one
    pointing at exact commit_sha.

    Returns dict with promotion result.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Delete existing prod branch if it exists
    try:
        api.delete_branch(repo_id=hf_repo, branch="prod")
        print(f"[INFO] Deleted existing 'prod' branch", file=sys.stderr)
    except Exception:
        print(f"[INFO] No existing 'prod' branch to delete", file=sys.stderr)

    # Create prod branch at the target revision
    api.create_branch(repo_id=hf_repo, branch="prod", revision=commit_sha)
    print(f"[INFO] Created 'prod' branch -> {commit_sha} (from {run_tag})",
          file=sys.stderr)

    return {
        "status": "promoted",
        "hf_repo": hf_repo,
        "run_tag": run_tag,
        "commit_sha": commit_sha,
        "prod_branch": "prod",
    }


# ---------------------------------------------------------------------------
# Local metrics scanning (for CI artifact-based flow)
# ---------------------------------------------------------------------------

def scan_local_metrics(metrics_dir: str) -> list[dict]:
    """Scan a local directory for run-*/metrics.json files.

    Expected structure:
      metrics_dir/
        run-1/metrics.json
        run-2/metrics.json
        ...

    Each metrics.json schema (input):
    {
      "perplexity": float,
      "token_accuracy": float,         // optional
      "opcode_accuracy": float,        // optional
      "assemble_pass_rate": float,     // optional
      "pass_at_1": float,              // optional
      "pass_at_5": float,              // optional
      "abi_lint_pass_rate": float,     // optional
      "seed": int,                     // logged for reproducibility
      "decoding_params": { ... }       // logged for reproducibility
    }
    """
    mdir = Path(metrics_dir)
    if not mdir.is_dir():
        return []

    runs = []
    for run_dir in sorted(mdir.iterdir()):
        mf = run_dir / "metrics.json"
        if mf.is_file():
            try:
                with open(mf) as f:
                    metrics = json.load(f)
                runs.append({
                    "tag": run_dir.name,
                    "commit_sha": None,
                    "metrics": metrics,
                })
            except (json.JSONDecodeError, IOError) as exc:
                print(f"[WARN] Skipping {mf}: {exc}", file=sys.stderr)
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="score_and_promote",
        description="Score model runs and promote the best to HF Hub 'prod' alias.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run: show best run
  python scripts/score_and_promote.py --hf-repo GNbros/rizz-v-model --auto-best

  # Promote specific run
  python scripts/score_and_promote.py --hf-repo GNbros/rizz-v-model \\
      --run-tag run-42 --promote --token $HF_TOKEN

  # Auto-promote best from local metrics
  python scripts/score_and_promote.py --metrics-dir ./outputs/metrics \\
      --hf-repo GNbros/rizz-v-model --auto-best --promote

Output JSON schema:
  {
    "action": "promote" | "dry-run",
    "selected_run": { "tag": str, "commit_sha": str, "metrics": {...} },
    "all_runs": [ ... ],
    "promotion_result": { ... } | null
  }
""",
    )
    p.add_argument("--hf-repo", required=True,
                    help="HF Hub repo id (e.g. GNbros/rizz-v-model)")
    p.add_argument("--run-tag",
                    help="Specific run tag to promote (e.g. run-42)")
    p.add_argument("--metrics-dir", default=None,
                    help="Local dir with run-*/metrics.json (alternative to HF scan)")
    p.add_argument("--auto-best", action="store_true",
                    help="Auto-select best run by lowest perplexity")
    p.add_argument("--promote", action="store_true",
                    help="Actually promote (otherwise dry-run)")
    p.add_argument("--token", default=None,
                    help="HF token (or set HF_TOKEN env var)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for determinism (default: 42)")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # -- Determinism --
    set_seed(args.seed)
    print(f"[INFO] Seed: {args.seed}", file=sys.stderr)

    token = args.token or os.environ.get("HF_TOKEN")

    # -- Collect runs --
    if args.metrics_dir:
        runs = scan_local_metrics(args.metrics_dir)
        print(f"[INFO] Found {len(runs)} runs in {args.metrics_dir}",
              file=sys.stderr)
    else:
        runs = list_run_revisions(args.hf_repo, token)
        print(f"[INFO] Found {len(runs)} run branches in {args.hf_repo}",
              file=sys.stderr)

    if not runs:
        print("[ERROR] No runs found.", file=sys.stderr)
        return 1

    # -- Print summary table --
    print("\n{:<15} {:>12} {:>12} {:>14}".format(
        "RUN TAG", "PERPLEXITY", "ASM PASS%", "COMMIT"), file=sys.stderr)
    print("-" * 57, file=sys.stderr)
    for r in runs:
        m = r.get("metrics") or {}
        ppl = m.get("perplexity", "N/A")
        asm = m.get("assemble_pass_rate", "N/A")
        sha = (r.get("commit_sha") or "local")[:10]
        ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else str(ppl)
        asm_str = f"{asm:.2%}" if isinstance(asm, (int, float)) else str(asm)
        print(f"{r['tag']:<15} {ppl_str:>12} {asm_str:>12} {sha:>14}",
              file=sys.stderr)
    print(file=sys.stderr)

    # -- Select run --
    selected = None

    if args.run_tag:
        selected = next((r for r in runs if r["tag"] == args.run_tag), None)
        if not selected:
            print(f"[ERROR] Run tag '{args.run_tag}' not found.", file=sys.stderr)
            return 1
    elif args.auto_best:
        selected = select_best_run(runs)
        if not selected:
            print("[ERROR] No runs with valid metrics for auto-select.",
                  file=sys.stderr)
            return 2
    else:
        print("[ERROR] Specify --run-tag or --auto-best.", file=sys.stderr)
        return 1

    print(f"[INFO] Selected: {selected['tag']}", file=sys.stderr)

    # -- Promote or dry-run --
    promotion_result = None

    if args.promote:
        if not selected.get("commit_sha"):
            # If from local metrics, we need to look up the HF commit
            hf_runs = list_run_revisions(args.hf_repo, token)
            hf_match = next(
                (r for r in hf_runs if r["tag"] == selected["tag"]), None
            )
            if not hf_match:
                print(f"[ERROR] Cannot find HF branch for '{selected['tag']}'.",
                      file=sys.stderr)
                return 1
            selected["commit_sha"] = hf_match["commit_sha"]

        promotion_result = promote_to_prod(
            hf_repo=args.hf_repo,
            commit_sha=selected["commit_sha"],
            run_tag=selected["tag"],
            token=token,
        )

    # -- Output JSON to stdout --
    output = {
        "action": "promote" if args.promote else "dry-run",
        "selected_run": selected,
        "all_runs": runs,
        "promotion_result": promotion_result,
        "seed": args.seed,
    }
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
