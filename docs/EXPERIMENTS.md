# Experiment Guide — RV32IM Assembly LLM Pipeline

This document defines the **recommended experiment matrix** and the
**metrics to watch** when evaluating base models for RISC-V assembly
code generation.

---

## 1. Recommended Experiment Matrix

### 1.1 Base Models

Start with these three tiers (small → medium) to cover different
architectures and tokenizer strategies:

| Model | Params | Architecture | Why include |
|---|---|---|---|
| `Salesforce/codegen-350M-multi` | 350 M | GPT-NeoX | Already validated on this repo; good baseline |
| `bigcode/starcoder2-3b` | 3 B | StarCoder2 | Strong code model, FIM-native tokenizer |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1 B | Llama-2 | Small Llama; tests whether chat-tuned models adapt |

> [!TIP]
> Adding a model is just a YAML file — see
> [README: Adding a New Base-Model](../README.md#adding-a-new-base-model-experiment).

### 1.2 Training Variants

For each base model, run these configurations:

| Variant | Config override | Purpose |
|---|---|---|
| **LoRA r=16** | `training.method=lora training.lora_r=16` | Default; fast & memory-efficient |
| **LoRA r=64** | `training.method=lora training.lora_r=64` | Higher-capacity adapter |
| **Full fine-tune** | `training.method=full` | Upper-bound quality (GPU-hungry) |

### 1.3 Dataset Mode Weights

Test at least two mixing strategies:

| Mix name | Causal | FIM | Block | Function | Hypothesis |
|---|---|---|---|---|---|
| **balanced** | 0.40 | 0.30 | 0.20 | 0.10 | Default; good all-round |
| **fim-heavy** | 0.20 | 0.50 | 0.20 | 0.10 | Better infill performance for IDE use |

Override in `configs/dataset.yaml` → `modes.weights` or via CLI:

```bash
python scripts/prepare_dataset.py build \
  --config configs/dataset.yaml \
  --override modes.weights.fim=0.5 modes.weights.causal=0.2
```

### 1.4 Full Matrix (MVP)

The MVP matrix is **3 models × 1 adapter × 1 mix = 3 runs**, completable
in ≈ 30 min on CI. The full matrix below is for thorough exploration:

| # | Model | Adapter | Mix | Expected CI time |
|---|---|---|---|---|
| 1 | codegen-350M | LoRA r=16 | balanced | ~10 min |
| 2 | starcoder2-3b | LoRA r=16 | balanced | ~20 min |
| 3 | TinyLlama-1.1B | LoRA r=16 | balanced | ~15 min |
| 4 | codegen-350M | LoRA r=16 | fim-heavy | ~10 min |
| 5 | starcoder2-3b | LoRA r=64 | balanced | ~25 min |
| 6 | codegen-350M | Full FT | balanced | ~40 min (GPU) |

> Run items **1–3 first** (the MVP set). Add rows 4–6 once you
> have a working comparison baseline.

---

## 2. Metrics to Watch

### 2.1 Primary Decision Metrics

These metrics determine which model gets **promoted to `prod`**:

| Metric | Script | Target (MVP) | Notes |
|---|---|---|---|
| **pass@1** | `eval_passk.py` | ≥ 0.40 | Must assemble with `riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32` |
| **pass@1 + ABI-lint** | `eval_passk.py --abi-lint` | ≥ 0.30 | Stricter; checks stack/ra/s-reg conventions |
| **assemble success rate** | `eval_assemble.py` | ≥ 0.55 | Fraction of generated samples that compile |

### 2.2 Diagnostic Metrics

Track these to understand *why* a model succeeds or fails:

| Metric | Script | What it tells you |
|---|---|---|
| **Perplexity** | `eval_text.py` | Overall language-model fit; lower = better |
| **Token accuracy** | `eval_text.py` | Exact next-token prediction rate |
| **Opcode accuracy** | `eval_text.py` | Accuracy restricted to instruction mnemonic tokens (`addi`, `lw`, etc.) |
| **Error histogram** | `eval_assemble.py` | Top error categories from GCC stderr (guides data augmentation) |
| **pass@5** | `eval_passk.py` | Ceiling potential with sampling; if pass@5 >> pass@1, lower temperature may help |

### 2.3 Metric Output File Paths

Every eval script writes a JSON file. Here is the complete map:

```
outputs/<model>/
├── metrics_text.json        # { perplexity, token_accuracy, opcode_accuracy }
├── metrics_assemble.json    # { assemble_success_rate, error_histogram }
└── metrics_passk.json       # { pass_at_1, pass_at_5, pass_at_1_abi_lint, pass_at_5_abi_lint }
```

All files include an `eval_config` block logging seed, temperature, and
decoding params for reproducibility.

### 2.4 Reading the Results

After a CI run completes, download the **artifacts** from the GitHub Actions
run page. Compare models side-by-side:

```bash
# Quick comparison (requires jq)
for model in codegen350m starcoder2_3b tinyllama; do
  echo "=== $model ==="
  jq '.pass_at_1, .assemble_success_rate' outputs/$model/metrics_passk.json outputs/$model/metrics_assemble.json
done
```

---

## 3. Deterministic Evaluation Checklist

Every experiment **must** follow these rules to ensure fair, reproducible
comparisons:

- [ ] `seed: 42` set in config and passed to all eval scripts via `--seed`
- [ ] Greedy decoding for text metrics: `temperature=0.0, top_p=1.0, do_sample=false`
- [ ] Sampling for pass@k: `temperature=0.8, top_p=0.95` (logged in output JSON)
- [ ] Same `max_seq_len` across all models in the same experiment row
- [ ] Same test split (`data/splits/test.jsonl`) — never re-split between runs
- [ ] GCC flags identical: `-march=rv32im -mabi=ilp32 -c` (no linking)
- [ ] Eval config block present in every output JSON

---

## 4. What to Do With Results

### Promotion

Promote the model with the highest **pass@1 + ABI-lint** that exceeds the
minimum threshold (default `0.40`):

```bash
python scripts/promote.py \
  --hub-repo GNbros/rizz-v-<model> \
  --source-branch run-<date> \
  --target-branch prod \
  --metrics-file outputs/<model>/metrics_passk.json \
  --min-pass-at-1 0.40
```

### Iteration Guidance

| Observation | Action |
|---|---|
| High perplexity, low token-acc | Model needs more training data or longer training |
| Token-acc OK but assemble fails | Opcode accuracy may be fine but operand / register formats are wrong — inspect error histogram |
| pass@5 >> pass@1 | Model has capability but greedy decode is suboptimal — try `temperature=0.2` |
| ABI-lint failures dominate | Add more function-level training samples with proper prologues/epilogues |
| One error type dominates histogram | Create targeted data augmentation for that pattern |

---

## 5. Future Experiments (Beyond MVP)

Once the MVP pipeline is stable, consider:

- **Larger models**: `codellama-7b`, `starcoder2-7b`
- **Instruction tuning**: Add natural-language instruction → assembly pairs
- **Multi-ISA transfer**: Pre-train on ARM/x86 assembly, fine-tune on RV32IM
- **Context-window scaling**: Compare 512 vs 1024 vs 2048 `max_seq_len`
- **Quantization**: Evaluate GPTQ/GGUF export for VS Code extension latency
