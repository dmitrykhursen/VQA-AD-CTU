# Thesis ref: Section 6.2 — LLaMA-Adapter V2 backend.
# Uses the local LLaMA-Adapter source from the DriveLM challenge repo.
# Not on HuggingFace — weights and base model live on the cluster.
# Use model name: OpenGVLab/llama_adapter_v2_multimodal7b
#
# Setup:
#   1. Clone DriveLM:  git clone https://github.com/OpenDriveLab/DriveLM third_party/DriveLM
#   2. Download LLaMA-7B weights into:
#        <adapter_src>/ckpts/llama_model_weights/
#   3. Download the LoRA checkpoint into:
#        <adapter_src>/ckpts/
#   4. Set env var:
#        export LLAMA_ADAPTER_SRC=/path/to/DriveLM/challenge/llama_adapter_v2_multimodal7b

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T

try:
    from torchvision.transforms import InterpolationMode
    _BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    _BICUBIC = 2  # fallback integer value

NEEDS_TILED_PIXELS = False  # LLaMA-Adapter does its own image preprocessing

# LLaMA-Adapter source — set via env var or override here for your cluster.
# Default assumes DriveLM was cloned into third_party/DriveLM (see setup above).
_DEFAULT_SRC = Path(__file__).resolve().parents[2] / "third_party" / "DriveLM" / "challenge" / "llama_adapter_v2_multimodal7b"
_ADAPTER_SRC = Path(os.environ.get("LLAMA_ADAPTER_SRC", str(_DEFAULT_SRC)))

# Known model → (checkpoint_path, llama_weights_dir)
MODEL_PATHS: dict[str, tuple[str, str]] = {
    "OpenGVLab/llama_adapter_v2_multimodal7b": (
        str(_ADAPTER_SRC / "ckpts" /
            "1bcbffc43484332672092e0024a8699a6eb5f558161aebf98a7c6b1db67224d1_LORA-BIAS-7B.pth"),
        str(_ADAPTER_SRC / "ckpts" / "llama_model_weights"),
    ),
}

# Image preprocessing — matches the original demo.py (CLIP normalisation, 224×224).
_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=_BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])


def _import_llama():
    """Import the llama package from the DriveLM challenge source."""
    if not _ADAPTER_SRC.exists():
        raise FileNotFoundError(
            f"LLaMA-Adapter source not found at {_ADAPTER_SRC}. "
            "Set the LLAMA_ADAPTER_SRC env var to the llama_adapter_v2_multimodal7b directory "
            "inside a local DriveLM clone (see module docstring for setup instructions)."
        )
    src = str(_ADAPTER_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import llama
        return llama
    except ImportError as exc:
        raise ImportError(
            f"Could not import 'llama' from {_ADAPTER_SRC}."
        ) from exc


def load(model_name: str, device: str = "cuda") -> tuple[Any, None]:
    """Load LLaMA-Adapter V2. Returns (model, None) — tokenizer is built into the model."""
    llama = _import_llama()

    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown LLaMA-Adapter model '{model_name}'. "
            f"Known names: {list(MODEL_PATHS)}"
        )
    ckpt, llama_dir = MODEL_PATHS[model_name]

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        model, _ = llama.load(ckpt, llama_dir, llama_type="7B", device=device)
    model.eval()
    return model, None


def batch_predict(
    bundle: tuple[Any, None],
    pixel_values: Any,          # unused — LLaMA-Adapter loads images itself
    questions: list[str],
    image_paths_per_sample: list[list[str]],
    *,
    num_patches_list: list[int] | None = None,  # unused, kept for interface parity
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Batched inference for LLaMA-Adapter V2 (stitched-collage mode).

    Uses only the first image path per sample (the pre-stitched collage).
    Preprocessing matches the original DriveLM demo: 224×224 resize + CLIP normalisation.
    """
    from PIL import Image

    llama = _import_llama()
    model, _ = bundle

    images = torch.stack([
        _TRANSFORM(Image.open(paths[0]).convert("RGB"))
        for paths in image_paths_per_sample
    ]).unsqueeze(1).to(device)  # (batch, 1, 3, H, W) — forward_visual iterates dim 1 as n_cameras

    prompts = [llama.format_prompt(q) for q in questions]

    with torch.inference_mode():
        results = model.generate(
            images,
            prompts,
            max_gen_len=max_new_tokens,
            temperature=0.0,
            top_p=0.75,
        )
    return [r.strip() for r in results]


def predict(
    bundle: tuple[Any, None],
    collage: Any,
    question: str,
    *,
    max_new_tokens: int = 512,
    device: str = "cuda",
    **_: Any,
) -> str:
    from PIL import Image

    llama = _import_llama()
    model, _ = bundle

    if isinstance(collage, (str, Path)):
        collage = Image.open(collage).convert("RGB")

    image = _TRANSFORM(collage).unsqueeze(0).to(device)
    prompt = llama.format_prompt(question)

    with torch.inference_mode():
        results = model.generate(image, [prompt], max_gen_len=max_new_tokens,
                                 temperature=0.0, top_p=0.75)
    return results[0].strip()
