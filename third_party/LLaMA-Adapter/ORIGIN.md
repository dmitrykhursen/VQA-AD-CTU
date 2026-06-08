# LLaMA-Adapter — Third-Party Origin

**Original:** https://github.com/OpenGVLab/LLaMA-Adapter

**License:** MIT

## What is used here

LLaMA-Adapter V2 is used as a multimodal baseline. The original source is not modified and is not committed to this repo.

Setup: clone or download [DriveLM](https://github.com/OpenDriveLab/DriveLM) and point the env var at the adapter directory:

```bash
export LLAMA_ADAPTER_SRC=/path/to/DriveLM/challenge/llama_adapter_v2_multimodal7b
```

The wrapper that loads and runs the model lives in [src/models/llama_adapterv2.py](../../src/models/llama_adapterv2.py).
See that file for checkpoint paths and full setup instructions.
