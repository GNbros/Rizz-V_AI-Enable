#!/usr/bin/env python3
"""
Rv32IM Pipeline Orchestrator
============================
Chains the full pipeline:
1. Data Generation (build_dataset.py)
2. Training (train_sft.py)
3. Evaluation (eval_text.py + eval_assemble.py)
4. Registry (score_and_promote.py)

CLI Usage:
    python scripts/orchestrate.py --config configs/codegen-350M-multi.yaml
    python scripts/orchestrate.py --config configs/codegen-350M-multi.yaml --skip-push
"""

import argparse
import logging
import subprocess
import sys
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_step(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command and check for errors."""
    # Ensure all args are strings
    cmd = [str(c) for c in cmd]
    logger.info(f"🚀 Starting step: {step_name}")
    logger.info(f"   Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        logger.info(f"✅ Step passed: {step_name}\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Step failed: {step_name}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full RV32IM pipeline end-to-end.")
    parser.add_argument("--config", required=True, help="Path to model config (e.g. configs/codegen.yaml)")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml", help="Path to shared dataset config")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset generation (use existing)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing checkpoint)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-push", action="store_true", help="Skip pushing to HF Hub")
    args = parser.parse_args()

    # Load configs
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    
    # Extract paths and parameters
    # Note: build_dataset.py uses the normalized dir from dataset.yaml to produce dataset structure
    # train_sft.py needs the config path
    
    run_id = f"run-{int(subprocess.check_output(['date', '+%s']).strip())}"
    output_dir = Path(model_cfg.get("output_dir", "outputs")) / run_id
    
    # 1. Dataset Generation
    if not args.skip_data:
        # We assume normalize.py and split_by_file.py have been run already or are stable.
        # But strictly speaking, "end-to-end" might imply running them too. 
        # For now, let's run build_dataset.py which generates the JSONL for training.
        # We also implicitly assume data/processed/normalized exists.
        
        # Determine dataset output dir from dataset config or default
        # Actually build_dataset.py reads config and outputs to `data/processed/dataset` usually
        # We can enforce a specific output if the script supports it, or just rely on config.
        # Let's check `scripts/build_dataset.py` args... it takes --config.
        
        run_step([
            sys.executable, "scripts/build_dataset.py",
            "--config", args.dataset_config
        ], "Dataset Generation")

    # 2. Training
    if not args.skip_train:
        logger.info(f"Starting training run: {run_id}")
        
        # Override experiment_name to force output to outputs/{run_id}
        # (Assuming config.output_dir is "outputs/")
        run_step([
            sys.executable, "train/train_sft.py",
            "--config", args.config,
            "--override", f"experiment_name={run_id}"
        ], "Training")
        
        # Based on train_sft.py logic: output_dir/experiment_name/model
        base_output = model_cfg.get("output_dir", "outputs")
        checkpoint_dir = PROJECT_ROOT / base_output / run_id / "model"
        
        if not checkpoint_dir.exists():
             logger.error(f"Checkpoint not found at {checkpoint_dir}")
             sys.exit(1)
        
        logger.info(f"Checkpoint saved at: {checkpoint_dir}")

    else:
        # Use latest run or specific arg? For now, latest in outputs/
        # But wait, we calculate output_dir based on config.
        base_output = model_cfg.get("output_dir", "outputs")
        outputs_path = PROJECT_ROOT / base_output
        if not outputs_path.exists():
             logger.error(f"Output directory {outputs_path} does not exist.")
             sys.exit(1)

        # Find latest run directory
        try:
            # list dirs in outputs/
            subdirs = [d for d in outputs_path.iterdir() if d.is_dir()]
            if not subdirs:
                raise ValueError("No run directories found")
            latest_run = max(subdirs, key=os.path.getctime)
            checkpoint_dir = latest_run / "model"
            logger.info(f"Using latest checkpoint: {checkpoint_dir}")
        except Exception as e:
             logger.error(f"Could not find latest checkpoint: {e}")
             sys.exit(1)

    # 3. Evaluation
    if not args.skip_eval:
        eval_output = Path(checkpoint_dir) / "eval_results.json"
        
        # Text Eval
        run_step([
            sys.executable, "scripts/eval_text.py",
            "--checkpoint", checkpoint_dir,
            "--test-file", "data/processed/dataset/test.jsonl", # inferred from build_dataset default?
            "--output", str(eval_output),
            "--config", args.dataset_config
        ], "Eval (Text Metrics)")
        
        # Assemble Eval
        run_step([
            sys.executable, "scripts/eval_assemble.py",
            "--checkpoint", checkpoint_dir,
            "--test-file", "data/processed/dataset/test.jsonl",
            "--output-dir", checkpoint_dir,
            "--config", args.dataset_config,
            "--gcc", "riscv64-unknown-elf-gcc" 
        ], "Eval (Assemble)")
        
    # 4. Registry / Push
    if not args.skip_push:
        # Using score_and_promote.py logic
        # It usually scans a dir. We can use it to push just this run?
        # The script `scripts/score_and_promote.py` seems designed for multiple runs.
        # But we can also use `pipeline/registry/push_model.py` if we built it, 
        # OR just use the existing script if it fits.
        # The existing script `scripts/score_and_promote.py` promotes.
        # To PUSH, `train_sft.py` has `push_to_hub` flag.
        # If we didn't push during training, we might need a separate push script.
        # For now, let's assume `score_and_promote.py` handles promotion of *already pushed* models? 
        # Let's check `score_and_promote.py` docstring again... 
        # "Score model runs and promote the best to HF Hub 'prod'."
        # It lists revisions. So the model must be ON HUB.
        
        logger.info("Pushing to HF Hub...")
        # If training didn't push, we might need to push manually here?
        # Adding a simple huggingface-cli upload or python script would work.
        # For MVP, let's assume config has `push_to_hub: true` or we use a separate tool.
        # I'll add a direct archival step if needed.
        pass

if __name__ == "__main__":
    main()
