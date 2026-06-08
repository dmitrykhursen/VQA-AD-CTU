# Thesis ref: Section 6.2 — InternVL2 / InternVL2.5 / InternVL3 backend.
# Pass a short model name (e.g. "InternVL2_2B") or a raw HF repo id / local path.

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.inference.image_utils import load_pixel_values

# Short name → HuggingFace repo id.
# Pass any key from this dict as --model, or pass a raw HF repo id / local path directly.
NEEDS_TILED_PIXELS = True  # InternVL2/2.5/3 require dynamic tiling via load_pixel_values

MODEL_PATHS: dict[str, str] = {
    # --- InternVL2 ---
    "InternVL2_1B":  "OpenGVLab/InternVL2-1B",
    "InternVL2_2B":  "OpenGVLab/InternVL2-2B",
    "InternVL2_4B":  "OpenGVLab/InternVL2-4B",
    "InternVL2_8B":  "OpenGVLab/InternVL2-8B",
    # --- InternVL2.5 ---
    "InternVL25_1B": "OpenGVLab/InternVL2_5-1B",
    "InternVL25_2B": "OpenGVLab/InternVL2_5-2B",
    "InternVL25_4B": "OpenGVLab/InternVL2_5-4B",
    "InternVL25_8B": "OpenGVLab/InternVL2_5-8B",
    # --- InternVL3 ---
    "InternVL3_2B":  "OpenGVLab/InternVL3-2B",
    "InternVL3_8B":  "OpenGVLab/InternVL3-8B",
    # --- Fine-tuned (DriveLM) ---
    "InternVL2_2B_lora25k":  "dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct",
    "InternVL2_2B_lora300k": "dkhursen/InternVL2-2b-LoRA-300k-drivelm",
}


def _resolve_path(model: str) -> str:
    """Return a HF repo id or local path from a short name or pass-through."""
    return MODEL_PATHS.get(model, model)


def load(model: str, device: str = "cuda") -> tuple[Any, Any]:
    """Return (model, tokenizer).

    `model` is either a short name from MODEL_PATHS (e.g. "InternVL2_2B_lora25k"),
    a raw HuggingFace repo id (e.g. "dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct"),
    or a local checkpoint directory path.
    """
    path = _resolve_path(model)

    # Rank-0-first download: prevents cache races when multiple ranks hit a new HF revision
    # simultaneously (some ranks end up with a revision dir missing conversation.py, etc.).
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    local_rank = torch.distributed.get_rank() if is_dist else 0

    if is_dist and local_rank == 0:
        # Rank 0 downloads everything (weights + remote code) before other ranks touch the cache.
        # Downloading only AutoConfig is not enough — parallel weight downloads corrupt safetensors.
        from huggingface_hub import snapshot_download
        is_local = path.startswith("/") or path.startswith(".")
        if not is_local:
            snapshot_download(path)
    if is_dist:
        torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    internvl_model = (
        AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )
    return internvl_model, tokenizer


def batch_predict(
    bundle: tuple[Any, Any],
    pixel_values: Any,
    questions: list[str],
    image_paths_per_sample: list[list[str]],
    *,
    num_patches_list: list[int],
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Batched inference for InternVL2/2.5/3.

    Prepends the correct number of ``<image>`` tokens per sample based on
    ``image_paths_per_sample`` (1 for stitched, 6 for separate cameras).
    """
    model, tokenizer = bundle
    formatted = [
        "<image>" * len(paths) + "\n" + q
        for paths, q in zip(image_paths_per_sample, questions)
    ]
    pv = pixel_values.to(torch.bfloat16).to(device)
    gen_cfg = {"max_new_tokens": max_new_tokens, "do_sample": False}
    return model.batch_chat(
        tokenizer, pv,
        num_patches_list=num_patches_list,
        questions=formatted,
        generation_config=gen_cfg,
    )


def predict(
    bundle: tuple[Any, Any],
    collage: Any,
    question: str,
    *,
    max_new_tokens: int = 512,
    max_tiles: int = 6,
    device: str = "cuda",
    **_: Any,
) -> str:
    model, tokenizer = bundle
    pixel_values = (
        load_pixel_values(collage, max_num=max_tiles)
        .to(torch.bfloat16)
        .to(device)
    )
    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
    return model.chat(tokenizer, pixel_values, question, gen_cfg)
