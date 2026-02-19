# Rizz-V AI Pipeline: Feature Overview & Architecture

## 🚀 high-Level Summary
The **Rizz-V AI Pipeline** is a newly re-architected, production-grade system designed to train and evaluate Large Language Models (LLMs) specifically for **RISC-V 32-bit Integer Multiply (RV32IM) Assembly**.

What was once a manual, notebook-based process is now a **modular, automated, and CI/CD-integrated workflow**.

---

## ✨ What's New? (Key Features)

### 1. 🧩 Modular & Model-Agnostic Architecture
*   **Old Way:** Hardcoded scripts for one specific model.
*   **New Way:** A flexible "plug-and-play" system. You can switch between **Salesforce Codegen (350M)**, **TinyLlama (1.1B)**, or **StarCoder** simply by changing a YAML configuration file. No code changes required.

### 2. 🤖 Advanced "Assemble-in-Context" Evaluation
*   **Old Way:** Only checked if the model's text looked somewhat like code (Perplexity).
*   **New Way:** We now compile the AI's output using a real **RISC-V GCC Toolchain**.
    *   **Pass@1 Metric:** What percentage of generated code *actually compiles*?
    *   **Opcode Accuracy:** Does it use valid RV32IM instructions?
    *   **Syntax Checking:** Categorizes errors (e.g., "unknown opcode", "missing ret").

### 3. 🧠 Smart Dataset Processing
*   **New Capability:** Supports **Fill-In-the-Middle (FIM)** training. This teaches the AI not just to write code from scratch, but to *complete* code in the middle of a file—crucial for IDE extensions.
*   **Metadata Tracking:** Every training sample tracks its source file, ensuring we can debug exactly where bad data came from.

### 4. ⚡️ Lightweight & Fast CI/CD
*   **Automated Quality Checks:** Every time code is pushed, a "Lightweight CI" runs instantly to check:
    *   **API Logic:** Is the web server working?
    *   **Math Verification:** Are our metric calculations correct?
    *   **Linter Rules:** Are the assembly validation rules accurate?
*   **Separation of Heavy Tasks:** Long-running training jobs are decoupled, triggered only when needed, saving cost and time.

---

## 🛠 Pipeline Workflow (The "Engine")

The pipeline operates in 4 distinct stages, orchestrated automatically:

### **Step 1: Data Engine (`build_dataset.py`)**
*   **Input:** Raw `.S` assembly files.
*   **Action:** Cleans, deduplicates, and splits data. Applies "FIM" masking for smarter training.
*   **Output:** `train.jsonl`, `test.jsonl` (JSON-Lines format).

### **Step 2: Training Engine (`train_sft.py`)**
*   **Input:** Clean Data + Base Model (e.g., Codegen-350M).
*   **Action:**
    *   Downloads base model.
    *   Injects custom RISC-V tokens (`<fim_prefix>`, `addi`, `jalr`).
    *   Fine-tunes using **LoRA (Low-Rank Adaptation)** for speed and efficiency.
*   **Output:** A fine-tuned adapter model optimized for RISC-V.

### **Step 3: Evaluation Engine (`eval_*.py`)**
*   **Action:**
    1.  **Text Eval:** Checks if the tokens look correct.
    2.  **Compiler Eval:** Runs `riscv64-unknown-elf-gcc` on generated outputs.
*   **Output:** a `metrics.json` report card (e.g., "95% Syntax Pass Rate").

### **Step 4: Inference API (`app.py`)**
*   **Action:** Serves the trained model via a fast REST API.
*   **Usage:** responding to IDE auto-complete requests in < 500ms.

---

## 📊 Configuration Strategy
We use **YAML-based configuration** to control experiments without touching code:

| Experiment Config | Base Model | Goal |
| :--- | :--- | :--- |
| `codegen_lora.yaml` | `Salesforce/codegen-350M` | **Balanced** (Speed/Quality default) |
| `tinyllama_full.yaml` | `TinyLlama-1.1B` | **Edge/Mobile** (Fast inference) |
| `starcoder_lora.yaml` | `StarCoder-1B` | **High Quality** (Smarter, slower) |

---

## 🏆 Success Metrics
*   **Pass@1 > 80%:** 80% of generated code must compile.
*   **Latency < 1000ms:** Auto-complete suggestions must appear instantly.
