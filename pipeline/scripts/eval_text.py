#!/usr/bin/env python3
"""
Text-Based Evaluation Metrics for RV32IM Assembly LLM
=====================================================
Computes perplexity, token accuracy, and opcode accuracy on a test split.

Input / Output Schema
---------------------
Input:
    --checkpoint  : path to HF model checkpoint directory
    --test-file   : path to test JSONL (one {"text": ..., "meta": ...} per line)
    --output      : path to write metrics.json
    --config      : path to configs/dataset.yaml (for opcode list)

Output (metrics.json):
    {
        "checkpoint": str,
        "test_file": str,
        "seed": int,
        "batch_size": int,
        "num_samples": int,
        "avg_loss": float,
        "perplexity": float,
        "token_accuracy": float,
        "opcode_accuracy": float,
        "opcode_total": int,
        "opcode_correct": int,
        "timestamp": str   (ISO-8601)
    }

CLI usage examples
------------------
    python -m scripts.eval_text \\
        --checkpoint runs/exp01/checkpoint-best \\
        --test-file data/processed/dataset/test.jsonl \\
        --output runs/exp01/metrics.json \\
        --seed 42 --batch-size 8

    python -m scripts.eval_text \\
        --checkpoint Salesforce/codegen-350M-multi \\
        --test-file data/processed/dataset/test.jsonl \\
        --output runs/baseline/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(config_path: str | None) -> dict[str, Any]:
    """Load configs/dataset.yaml and return the parsed dict."""
    if config_path is None:
        default = Path(__file__).resolve().parent.parent / "configs" / "dataset.yaml"
        if default.exists():
            config_path = str(default)
        else:
            return {}
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def get_opcodes(config: dict[str, Any]) -> set[str]:
    """Return the canonical RV32IM opcode set from config."""
    opcodes = config.get("rv32im_opcodes", [])
    if not opcodes:
        # Fallback minimal set
        opcodes = [
            "add", "sub", "and", "or", "xor", "sll", "srl", "sra",
            "slt", "sltu", "addi", "andi", "ori", "xori", "slti",
            "sltiu", "slli", "srli", "srai", "lw", "sw", "lb", "sb",
            "lh", "sh", "lbu", "lhu", "lui", "auipc", "jal", "jalr",
            "beq", "bne", "blt", "bge", "bltu", "bgeu",
            "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
            "nop", "li", "la", "mv", "ret", "j", "jr", "call", "tail",
            "not", "neg", "seqz", "snez", "sltz", "sgtz",
            "fence", "ecall", "ebreak",
        ]
    return {op.lower() for op in opcodes}


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file, one JSON object per line."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Dataset wrapper ──────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Simple dataset that returns tokenized text from JSONL records."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.texts = [r["text"] for r in records]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        return input_ids


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    max_len = max(ids.size(0) for ids in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ids in enumerate(batch):
        padded[i, :ids.size(0)] = ids
        attention_mask[i, :ids.size(0)] = 1
    return {"input_ids": padded, "attention_mask": attention_mask}


# ── Metric computation ──────────────────────────────────────────────────────

_LINE_TOKEN_RE = re.compile(r"^\s*([a-zA-Z_]\w*)")


def compute_opcode_accuracy(
    texts: list[str],
    model,
    tokenizer,
    valid_opcodes: set[str],
    device: torch.device,
    max_length: int = 512,
) -> dict[str, int]:
    """
    For each assembly line in each text, extract the opcode (first token),
    compare the model's greedy prediction of that position against ground-truth.

    Returns dict with keys: opcode_total, opcode_correct.
    """
    total = 0
    correct = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # (1, seq_len, vocab)

        input_ids = enc["input_ids"][0]
        # Greedy predicted tokens (shifted by 1)
        pred_ids = logits[0].argmax(dim=-1)  # (seq_len,)

        # Decode token by token
        decoded_input = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Find positions at start of lines that are opcodes
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        lines = full_text.split("\n")

        char_pos = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(".") or stripped.startswith("#"):
                char_pos += len(line) + 1
                continue

            m = _LINE_TOKEN_RE.match(stripped)
            if not m:
                char_pos += len(line) + 1
                continue

            first_token = m.group(1).lower()
            if first_token in valid_opcodes:
                # This line starts with a valid opcode – count it
                total += 1

                # Find the token position in the tokenizer output
                prefix = full_text[:char_pos + line.index(stripped)]
                prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))

                if prefix_len < len(pred_ids):
                    pred_token_id = pred_ids[prefix_len].item()
                    pred_str = tokenizer.decode([pred_token_id]).strip().lower()
                    # Check if predicted token matches the opcode
                    if pred_str == first_token or pred_str.startswith(first_token):
                        correct += 1

            char_pos += len(line) + 1

    return {"opcode_total": total, "opcode_correct": correct}


def evaluate_text_metrics(
    checkpoint: str,
    test_file: str,
    output_path: str,
    config_path: str | None = None,
    seed: int = 42,
    batch_size: int = 8,
    max_length: int = 512,
) -> dict[str, Any]:
    """
    End-to-end evaluation: loads model, computes loss/perplexity,
    token accuracy, and opcode accuracy.

    Returns the metrics dict and writes it to output_path.
    """
    # Determinism
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = load_config(config_path)
    valid_opcodes = get_opcodes(config)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load data
    records = load_jsonl(test_file)
    dataset = TextDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # ── Pass 1: loss + token accuracy ────────────────────────────────────
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Token accuracy: compare shifted logits vs labels
            logits = outputs.logits[:, :-1, :]        # (B, S-1, V)
            labels = input_ids[:, 1:]                  # (B, S-1)
            mask = attention_mask[:, 1:].bool()        # (B, S-1)

            preds = logits.argmax(dim=-1)              # (B, S-1)
            correct_tokens += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    token_acc = correct_tokens / max(total_tokens, 1)

    # ── Pass 2: opcode accuracy ──────────────────────────────────────────
    texts = [r["text"] for r in records]
    opcode_stats = compute_opcode_accuracy(texts, model, tokenizer, valid_opcodes, device, max_length)
    opcode_acc = opcode_stats["opcode_correct"] / max(opcode_stats["opcode_total"], 1)

    # ── Assemble metrics ─────────────────────────────────────────────────
    metrics = {
        "checkpoint": checkpoint,
        "test_file": test_file,
        "seed": seed,
        "batch_size": batch_size,
        "num_samples": len(records),
        "avg_loss": round(avg_loss, 6),
        "perplexity": round(perplexity, 6),
        "token_accuracy": round(token_acc, 6),
        "opcode_accuracy": round(opcode_acc, 6),
        "opcode_total": opcode_stats["opcode_total"],
        "opcode_correct": opcode_stats["opcode_correct"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eval_text",
        description="Compute text-based evaluation metrics for RV32IM assembly models.",
        epilog="""
CLI usage examples:
  python -m scripts.eval_text \\
      --checkpoint runs/exp01/checkpoint-best \\
      --test-file data/processed/dataset/test.jsonl \\
      --output runs/exp01/metrics.json

  python -m scripts.eval_text \\
      --checkpoint Salesforce/codegen-350M-multi \\
      --test-file data/processed/dataset/test.jsonl \\
      --output runs/baseline/metrics.json \\
      --seed 42 --batch-size 4
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="HF model checkpoint path or name")
    parser.add_argument("--test-file", required=True, help="Path to test.jsonl")
    parser.add_argument("--output", required=True, help="Path to write metrics.json")
    parser.add_argument("--config", default=None, help="Path to configs/dataset.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism (default: 42)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length (default: 512)")

    args = parser.parse_args()

    print(f"[eval_text] checkpoint={args.checkpoint}")
    print(f"[eval_text] test_file={args.test_file}")
    print(f"[eval_text] seed={args.seed}  batch_size={args.batch_size}")

    metrics = evaluate_text_metrics(
        checkpoint=args.checkpoint,
        test_file=args.test_file,
        output_path=args.output,
        config_path=args.config,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"\n[eval_text] Results written to {args.output}")
    print(f"  loss={metrics['avg_loss']:.4f}  ppl={metrics['perplexity']:.4f}")
    print(f"  token_acc={metrics['token_accuracy']:.4f}")
    print(f"  opcode_acc={metrics['opcode_accuracy']:.4f} ({metrics['opcode_correct']}/{metrics['opcode_total']})")


if __name__ == "__main__":
    main()
