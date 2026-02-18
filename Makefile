# Makefile for RV32IM Assembly LLM Pipeline

.PHONY: help install normalize build-dataset train eval push run-pipeline test-smoke clean

PYTHON := python3

help:
	@echo "RV32IM Pipeline Makefile"
	@echo "========================"
	@echo "Targets:"
	@echo "  install        : Install dependencies"
	@echo "  normalize      : Run normalize_asm.py"
	@echo "  build-dataset  : Run build_dataset.py"
	@echo "  train          : Run train_sft.py"
	@echo "  eval           : Run eval_text.py + eval_assemble.py"
	@echo "  push           : Run score_and_promote.py"
	@echo "  run-pipeline   : Run full end-to-end pipeline (orchestrate.py)"
	@echo "  test-smoke     : Run E2E smoke test (pytest)"
	@echo "  clean          : Remove output directories"

install:
	pip install -r requirements.txt

# Individual steps wrapping scripts (examples)
normalize:
	$(PYTHON) -m scripts.normalize_asm --config configs/dataset.yaml

build-dataset:
	$(PYTHON) -m scripts.build_dataset --config configs/dataset.yaml

# Train requires a specific Model Config
train:
	@if [ -z "$(CONFIG)" ]; then echo "Error: CONFIG argument required (e.g. make train CONFIG=configs/codegen.yaml)"; exit 1; fi
	$(PYTHON) -m train.train_sft --config $(CONFIG)

# Orchestrator
run-pipeline:
	@if [ -z "$(CONFIG)" ]; then echo "Error: CONFIG argument required"; exit 1; fi
	$(PYTHON) scripts/orchestrate.py --config $(CONFIG) $(if $(SKIP_PUSH),--skip-push,)

# Smoke Test
test-smoke:
	pytest tests/test_e2e_smoke.py -v

clean:
	rm -rf outputs/ runs/ data/processed/dataset data/processed/splits tests/fixtures/output
