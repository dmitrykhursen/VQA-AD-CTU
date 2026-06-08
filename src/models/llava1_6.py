# Thesis ref: Section 6.2 — LLaVA-NeXT (1.6) backend.
# Requires transformers >= 4.40 and flash-attn >= 2.x.
# Typical repo: llava-hf/llava-v1.6-mistral-7b-hf

from __future__ import annotations

from typing import Any

import torch

NEEDS_TILED_PIXELS = False  # LLaVA reloads images via its own processor; InternVL tiles are unused


def load(model_path: str, device: str = "cuda") -> tuple[Any, Any]:
    """Return (model, processor). model_path is a HF repo id or local dir.

    flash_attention_2 is used when available; requires float16 (not bfloat16).
    The processor's padding_side is set to 'left' so batch generation aligns
    new tokens at the end of every sequence.
    """
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

    processor = LlavaNextProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
    model = (
        LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        .eval()
        .to(device)
    )
    return model, processor


def predict(
    bundle: tuple[Any, Any],
    collage: Any,
    question: str,
    *,
    max_new_tokens: int = 512,
    device: str = "cuda",
    **_: Any,
) -> str:
    model, processor = bundle
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=collage, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


def batch_predict(
    bundle: tuple[Any, Any],
    pixel_values: Any,  # unused — LLaVA uses its own processor on raw images
    questions: list[str],
    image_paths_per_sample: list[list[str]],
    *,
    num_patches_list: list[int] | None = None,  # unused, kept for interface parity with InternVL
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Batched inference for LLaVA-NeXT (stitched-collage mode).

    The processor's chat template inserts image tokens automatically via
    the ``{"type": "image"}`` content block — no manual ``<image>`` prepending.
    Only the first path per sample is used (one stitched collage per QA).
    Left padding (set in load()) ensures generated tokens sit at position
    padded_input_len onward for every sample in the batch.
    """
    from PIL import Image

    model, processor = bundle
    images = [Image.open(paths[0]).convert("RGB") for paths in image_paths_per_sample]
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
            add_generation_prompt=True,
        )
        for q in questions
    ]
    inputs = processor(
        images=images, text=prompts, return_tensors="pt", padding=True
    ).to(device)
    padded_input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return [
        processor.decode(out[padded_input_len:], skip_special_tokens=True).strip()
        for out in output_ids
    ]
