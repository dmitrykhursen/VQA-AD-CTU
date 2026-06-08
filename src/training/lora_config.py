"""
LoRA configuration for InternVL2-2B fine-tuning.

InternVL applies LoRA via its own PEFT integration when ``--use_llm_lora <rank>``
is passed to ``internvl/train/internvl_chat_finetune.py``.  The alpha is
hardcoded inside InternVL as ``2 × rank``; all linear layers of the LLM are
targeted (wqkv, wo, w1, w2, w3).  The ViT backbone and MLP projector are kept
frozen.

This module documents those settings as a dataclass and exposes a helper that
produces the CLI argument fragments consumed by :mod:`src.training.train`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """LoRA hyperparameters as reported in the thesis (Section 6.1.3).

    Attributes
    ----------
    rank:
        LoRA rank *r*.  InternVL hardcodes alpha = 2 × rank (= 32 here).
    dropout:
        LoRA dropout applied to the adapter layers.
    freeze_backbone:
        Keep the InternVL ViT backbone frozen (no LoRA on vision encoder).
    freeze_mlp:
        Keep the MLP projector frozen.
    freeze_llm:
        Freeze the base LLM weights; only the injected LoRA adapters are
        updated during training.
    """

    rank: int = 16
    # alpha is set by InternVL internals to 2 * rank → 32
    dropout: float = 0.05
    freeze_backbone: bool = True
    freeze_mlp: bool = True
    freeze_llm: bool = True  # only LoRA adapters are trainable

    def cli_args(self) -> List[str]:
        """Return CLI flags to pass to ``internvl_chat_finetune.py``."""
        return [
            "--use_llm_lora",    str(self.rank),
            "--freeze_llm",      str(self.freeze_llm),
            "--freeze_mlp",      str(self.freeze_mlp),
            "--freeze_backbone", str(self.freeze_backbone),
        ]


# Default config used for all experiments in the thesis
DEFAULT_LORA = LoRAConfig()
