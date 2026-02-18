# Breaking Changes Log

> **Rule H3**: Any agent changing a cross-boundary interface MUST append an entry here.
> QA/Reviewer blocks merge of any PR that modifies a shared interface without a corresponding entry.

---

## 2026-02-18 — PM — T1.1
- **Changed**: `eval_decoding` section in `configs/dataset.yaml` renamed to `eval`; fields `decoding` and `num_samples_pass_k` added; `temperature` default changed from `0.8` → `0.0`; `top_p` default changed from `0.95` → `1.0`
- **Affects**: Eval Engineer (any script reading `eval_decoding`), Training Engineer (if referencing eval params)
- **Migration**: Replace `cfg['eval_decoding']` → `cfg['eval']` in all scripts
