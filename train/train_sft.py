#!/usr/bin/env python3
"""
Model-Agnostic SFT Training Script for RV32IM Assembly LLM Pipeline.

Loads any HuggingFace causal-LM via AutoTokenizer/AutoModelForCausalLM,
adds special tokens from the shared registry (configs/dataset.yaml),
and fine-tunes with LoRA (default) or full fine-tune.

CLI usage:
    # Basic training
    python train/train_sft.py --config configs/experiments/codegen_lora.yaml

    # Override specific params at CLI
    python train/train_sft.py --config configs/train.yaml \\
        --override base_model_id=Salesforce/codegen-350M-multi \\
        --override training.num_epochs=1

    # Smoke test (CI)
    python train/train_sft.py --config configs/experiments/smoke_test.yaml

Input:
    --config  : YAML config file (see configs/train.yaml for schema)
    Dataset   : JSONL files from config.dataset.files[].path
                Each line: {"text": "...", "meta": {...}}
    Token reg : configs/dataset.yaml (shared special tokens)

Output (written to runs/<experiment_name>/):
    model/            : Saved model (merged LoRA if applicable)
    tokenizer/        : Tokenizer with added special tokens
    run_metadata.json : Config snapshot, metrics, timestamps, hardware info
    checkpoints/      : Intermediate checkpoints (per save_strategy)
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_sft")

# ---------------------------------------------------------------------------
# Project root (two levels up from train/train_sft.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Config helpers
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config and resolve paths relative to project root."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dotted key=value overrides to a nested config dict.

    Example: training.num_epochs=1  →  cfg["training"]["num_epochs"] = 1
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov}")
        key, val = ov.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Attempt type coercion
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[parts[-1]] = val
        logger.info(f"Override applied: {key} = {val}")
    return cfg


def load_token_registry(cfg: dict) -> dict:
    """Load shared token registry from configs/dataset.yaml."""
    registry_path = cfg.get("token_registry", "configs/dataset.yaml")
    full_path = PROJECT_ROOT / registry_path
    if not full_path.exists():
        logger.warning(
            f"Token registry not found at {full_path}, using empty registry"
        )
        return {}
    with open(full_path, "r") as f:
        registry = yaml.safe_load(f)
    return registry


# ============================================================================
# Special tokens
# ============================================================================

def get_special_tokens(registry: dict) -> list[str]:
    """Extract the list of special token strings from the registry."""
    st = registry.get("special_tokens", {})
    tokens = []
    for key, val in st.items():
        if key == "version":
            continue
        if isinstance(val, str) and val.startswith("<"):
            tokens.append(val)
    return tokens


def add_special_tokens_to_tokenizer(tokenizer, tokens: list[str]) -> list[str]:
    """Add missing special tokens to tokenizer. Returns list of newly added tokens."""
    existing = set(tokenizer.all_special_tokens) | set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in tokens if t not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Added {len(new_tokens)} special tokens: {new_tokens}")
    else:
        logger.info("All special tokens already present in tokenizer")
    return new_tokens


def resize_embeddings(model, tokenizer):
    """Resize model embeddings to match tokenizer vocab; mean-init new embeddings."""
    old_size = model.get_input_embeddings().weight.shape[0]
    new_size = len(tokenizer)
    if new_size > old_size:
        model.resize_token_embeddings(new_size)
        # Mean-init new embeddings for better starting point
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean_emb = emb[:old_size].mean(dim=0)
            emb[old_size:] = mean_emb
        logger.info(
            f"Resized embeddings: {old_size} → {new_size} "
            f"(mean-init for {new_size - old_size} new tokens)"
        )
    return old_size, new_size


# ============================================================================
# Dataset loading
# ============================================================================

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, extracting the 'text' field."""
    records = []
    full_path = PROJECT_ROOT / path
    with open(full_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append({"text": obj["text"]})
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping line {i+1} in {path}: {e}")
    logger.info(f"Loaded {len(records)} samples from {path}")
    return records


def load_dataset_from_config(cfg: dict, split: str = "train") -> Dataset:
    """Load and (optionally) weight-mix multiple JSONL files."""
    if split == "eval":
        eval_path = cfg["dataset"].get("eval_file")
        if not eval_path:
            logger.warning("No eval_file specified; skipping eval dataset")
            return None
        records = load_jsonl(eval_path)
        return Dataset.from_list(records) if records else None

    # Training: support weighted mixing
    all_records = []
    for file_spec in cfg["dataset"]["files"]:
        path = file_spec["path"]
        weight = file_spec.get("weight", 1.0)
        records = load_jsonl(path)
        # Simple weighting: duplicate records by integer weight
        int_weight = max(1, int(weight))
        all_records.extend(records * int_weight)

    return Dataset.from_list(all_records)


# ============================================================================
# Tokenization
# ============================================================================

def tokenize_fn(examples, tokenizer, max_length):
    """Tokenize a batch of texts for causal LM training."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    # Labels = input_ids, with padding masked
    labels = []
    pad_token_id = tokenizer.pad_token_id
    for ids in tokenized["input_ids"]:
        label = [(-100 if token_id == pad_token_id else token_id) for token_id in ids]
        labels.append(label)
    tokenized["labels"] = labels
    return tokenized


# ============================================================================
# LoRA setup
# ============================================================================

def setup_lora(model, cfg: dict):
    """Wrap model with LoRA adapters via PEFT."""
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = cfg["lora"]
    task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_str, TaskType.CAUSAL_LM)

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ============================================================================
# Hardware detection
# ============================================================================

def get_hardware_info() -> dict:
    """Collect hardware info for run metadata."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 2
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
    else:
        info["device"] = "cpu"
    return info


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model-agnostic SFT training for RV32IM Assembly LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train/train_sft.py --config configs/experiments/smoke_test.yaml
  python train/train_sft.py --config configs/experiments/codegen_lora.yaml
  python train/train_sft.py --config configs/train.yaml \\
      --override base_model_id=Salesforce/codegen-350M-multi
        """,
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values: key.subkey=value (repeatable)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load config
    # -----------------------------------------------------------------------
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    experiment_name = cfg.get("experiment_name", "default")
    train_cfg = cfg.get("training", {})
    seed = train_cfg.get("seed", 42)

    # Determinism
    set_seed(seed)
    logger.info(f"Experiment: {experiment_name} | Seed: {seed}")

    # -----------------------------------------------------------------------
    # 2. Load token registry
    # -----------------------------------------------------------------------
    registry = load_token_registry(cfg)
    special_tokens = get_special_tokens(registry)
    logger.info(f"Token registry loaded: {len(special_tokens)} special tokens")

    # -----------------------------------------------------------------------
    # 3. Load model and tokenizer
    # -----------------------------------------------------------------------
    model_id = cfg["base_model_id"]
    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if train_cfg.get("fp16") else torch.float32,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token = eos_token ({tokenizer.eos_token})")

    # -----------------------------------------------------------------------
    # 4. Add special tokens + resize embeddings
    # -----------------------------------------------------------------------
    vocab_before = len(tokenizer)
    new_tokens = add_special_tokens_to_tokenizer(tokenizer, special_tokens)
    vocab_after = len(tokenizer)
    old_emb, new_emb = resize_embeddings(model, tokenizer)

    # -----------------------------------------------------------------------
    # 5. Apply LoRA (if enabled)
    # -----------------------------------------------------------------------
    lora_cfg = cfg.get("lora", {})
    training_mode = "full"
    lora_params = {}

    if lora_cfg.get("enabled", True):
        training_mode = "lora"
        model = setup_lora(model, cfg)
        lora_params = {
            "r": lora_cfg.get("r"),
            "alpha": lora_cfg.get("alpha"),
            "dropout": lora_cfg.get("dropout"),
            "target_modules": lora_cfg.get("target_modules"),
        }
        logger.info(f"LoRA enabled: r={lora_params['r']}, alpha={lora_params['alpha']}")
    else:
        logger.info("Full fine-tune mode (LoRA disabled)")

    # -----------------------------------------------------------------------
    # 6. Load and tokenize datasets
    # -----------------------------------------------------------------------
    max_length = train_cfg.get("max_length", 512)

    train_ds = load_dataset_from_config(cfg, split="train")
    train_ds = train_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train",
    )

    eval_ds = load_dataset_from_config(cfg, split="eval")
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda ex: tokenize_fn(ex, tokenizer, max_length),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval",
        )

    logger.info(
        f"Dataset: {len(train_ds)} train samples"
        + (f", {len(eval_ds)} eval samples" if eval_ds else "")
    )

    # -----------------------------------------------------------------------
    # 7. Setup training arguments
    # -----------------------------------------------------------------------
    run_dir = PROJECT_ROOT / cfg.get("output_dir", "runs/") / experiment_name
    checkpoint_dir = run_dir / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        num_train_epochs=train_cfg.get("num_epochs", 3),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        eval_strategy=train_cfg.get("eval_strategy", "epoch") if eval_ds else "no",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        report_to="none",
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        remove_unused_columns=False,
    )

    # -----------------------------------------------------------------------
    # 8. Train
    # -----------------------------------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    logger.info("Starting training...")
    train_result = trainer.train()
    elapsed = time.time() - start_time
    completed_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        f"Training completed in {elapsed:.1f}s | "
        f"Final train loss: {train_result.training_loss:.4f}"
    )

    # -----------------------------------------------------------------------
    # 9. Evaluate
    # -----------------------------------------------------------------------
    eval_results = {}
    if eval_ds is not None:
        eval_results = trainer.evaluate()
        logger.info(f"Eval loss: {eval_results.get('eval_loss', 'N/A')}")

    # -----------------------------------------------------------------------
    # 10. Save model + tokenizer
    # -----------------------------------------------------------------------
    model_dir = run_dir / "model"
    tokenizer_dir = run_dir / "tokenizer"

    # Merge LoRA weights if applicable
    if training_mode == "lora":
        logger.info("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(model_dir))
    else:
        model.save_pretrained(str(model_dir))

    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info(f"Model saved to {model_dir}")
    logger.info(f"Tokenizer saved to {tokenizer_dir}")

    # -----------------------------------------------------------------------
    # 11. Write run metadata
    # -----------------------------------------------------------------------
    metadata = {
        "experiment_name": experiment_name,
        "base_model_id": model_id,
        "config_snapshot": cfg,
        "seed": seed,
        "deterministic": True,
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds": round(elapsed, 2),
        "final_train_loss": round(train_result.training_loss, 6),
        "final_eval_loss": round(eval_results.get("eval_loss", 0), 6)
        if eval_results
        else None,
        "total_steps": train_result.global_step,
        "hardware": get_hardware_info(),
        "special_tokens_added": new_tokens,
        "vocab_size_before": vocab_before,
        "vocab_size_after": vocab_after,
        "training_mode": training_mode,
        "lora_params": lora_params if training_mode == "lora" else None,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds) if eval_ds else 0,
    }

    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"  Experiment   : {experiment_name}")
    logger.info(f"  Model        : {model_id}")
    logger.info(f"  Mode         : {training_mode}")
    logger.info(f"  Train loss   : {train_result.training_loss:.4f}")
    if eval_results:
        logger.info(f"  Eval loss    : {eval_results.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"  Steps        : {train_result.global_step}")
    logger.info(f"  Duration     : {elapsed:.1f}s")
    logger.info(f"  Output       : {run_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
