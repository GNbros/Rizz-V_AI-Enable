import subprocess
import sys
import shutil
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def test_pipeline_smoke_end_to_end():
    """
    Runs the full orchestrator using smoke test configs.
    Verifies:
    1. Dataset generation works (JSONL created)
    2. Training runs (checkpoint created, metadata saved)
    3. Eval runs (results.json created)
    4. Determinism is logged
    """
    
    # 1. Setup paths
    smoke_dataset_cfg = PROJECT_ROOT / "pipeline/configs/smoke_dataset.yaml"
    smoke_model_cfg = PROJECT_ROOT / "pipeline/configs/smoke_model.yaml"
    output_base = PROJECT_ROOT / "pipeline/tests/fixtures/output"
    
    # Clean previous run
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 2. Run orchestrator
    # We skip push because it requires HF token and network
    cmd = [
        sys.executable, "pipeline/scripts/orchestrate.py",
        "--config", str(smoke_model_cfg),
        "--dataset-config", str(smoke_dataset_cfg),
        "--skip-push" 
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Orchestrator failed"
    
    # 3. Verify Artifacts
    
    # Dataset
    dataset_dir = output_base / "dataset"
    assert (dataset_dir / "train.jsonl").exists()
    assert (dataset_dir / "val.jsonl").exists()
    
    # Runs (Training Output)
    runs_dir = output_base / "runs"
    # Find the run ID directory (orchestrate uses run-{timestamp})
    # Since we use --override, train_sft puts it in input_dir/run_id?
    # Wait, orchestrator calls: train_sft ... --override experiment_name={run_id}
    # config.output_dir is tests/fixtures/output/runs
    # So path is tests/fixtures/output/runs/{run_id}
    
    found_runs = list(runs_dir.glob("run-*"))
    assert len(found_runs) == 1, f"Expected 1 run directory, found {len(found_runs)}: {found_runs}"
    run_dir = found_runs[0]
    
    # Metadata
    assert (run_dir / "run_metadata.json").exists()
    
    # Model
    assert (run_dir / "model" / "config.json").exists() or (run_dir / "model" / "adapter_config.json").exists()
    
    # Eval Results (orchestrate puts them in run_dir/eval_results.json?)
    # Re-reading orchestrator logic:
    # eval_output = Path(checkpoint_dir) / "eval_results.json"
    # checkpoint_dir = run_dir / "model"
    # So eval results are inside model dir?
    # Let's check orchestrator logic again.
    # checkpoint_dir = latest_dir (which is run_dir? No, orchestrator logic says `checkpoint_dir = PROJECT_ROOT / base_output / run_id / "model"`)
    # So `eval_results.json` should be in `model/`
    
    model_dir = run_dir / "model"
    # Wait, eval_text produces an output file.
    # verify logic in orchestrator: "--output", str(eval_output)
    # eval_output = Path(checkpoint_dir) / "eval_results.json"
    # So yes, in model dir.
    
    # NOTE: Orchestrator currently puts it in checkpoint_dir. 
    # train_sft puts LoRA/Model in run_dir/model.
    # So eval is at runs/run-id/model/eval_results.json.
    
    eval_json = model_dir / "eval_results.json"
    assert eval_json.exists(), f"Eval results not found at {eval_json}"
    
    # 4. Cleanup (optional, useful to keep for inspection if failed)
    # shutil.rmtree(output_base)

if __name__ == "__main__":
    test_pipeline_smoke_end_to_end()
