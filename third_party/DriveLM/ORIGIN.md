# DriveLM — Third-Party Origin

**Original:** https://github.com/OpenDriveLab/DriveLM
**License:** Apache 2.0

## What is committed here

Only this ORIGIN.md. The DriveLM source is not bundled in the repo.

## What was adapted

- Dataset extraction and format conversion scripts were used as-is during data preparation.
- The evaluation scripts (`evaluation.py`, `evaluation_extended.py`, `gpt_eval.py`) were modified and now live in `src/evaluation/` — see that directory for what changed.
- The dataset class was adapted and lives in `src/datasets/`.

## When you need to clone DriveLM locally

Only if running the LLaMA-Adapter V2 baseline. Clone into `third_party/DriveLM` and set:

```bash
export LLAMA_ADAPTER_SRC="$PWD/third_party/DriveLM/challenge/llama_adapter_v2_multimodal7b"
```

See [third_party/LLaMA-Adapter/ORIGIN.md](../LLaMA-Adapter/ORIGIN.md) for full setup.
