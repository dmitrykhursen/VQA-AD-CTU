# InternVL – Third-Party Dependency

This directory holds the InternVL repository used for Stage 4 (LoRA fine-tuning of InternVL2-2B).
The actual code is **not** committed here; clone it as described below.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/OpenGVLab/InternVL.git third_party/InternVL
cd third_party/InternVL
# Tested at commit b3f38dc on branch main
git checkout b3f38dc
```

### 2. Install the training extras

```bash
cd third_party/InternVL/internvl_chat
pip install -e ".[train]"
```

### 3. Install FlashAttention

```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

---

## Modifications from the original

The following files were modified for this project.  
The modified versions live under `src/training/` and are forwarded via `PYTHONPATH`; the upstream files
inside `internvl_chat/` are **not** patched.

| File (relative to `internvl_chat/`) | Change |
|--------------------------------------|--------|
| `internvl/train/internvl_chat_finetune.py` | Added `--eval_meta_path`, `--inference_batch_size`, and `--loss_reduction` CLI arguments; added `run_drivelm_metrics` custom evaluation callback that computes DriveLM-specific metrics at the end of each epoch. |
| `tools/merge_lora.py` | Added `--all_folder` batch-merge mode that finds every `checkpoint-*` subdirectory inside a given work directory and merges each one into `<work_dir>_merged/<checkpoint_name>/`. The adapted version lives at `src/training/merge_lora.py`. |

---

## Working directory for training

All training launches (`torchrun` / `scripts/04_finetune.sh`) set the working directory to:

```
third_party/InternVL/internvl_chat/
```

and set `PYTHONPATH` to include both `internvl_chat/` and the project root so that project-level modules
(`src/…`) can be imported alongside InternVL internals.
