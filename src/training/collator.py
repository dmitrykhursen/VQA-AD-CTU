"""
Data collator for InternVL2-2B multi-modal training batches.

InternVL provides ``concat_pad_data_collator`` (in
``internvl/patch/__init__.py``) which handles:

- Packing tokenised QA pairs with correct padding and attention masks
- Stacking variable-size image tile tensors (dynamic resolution)
- Aligning ``<IMG_CONTEXT>`` token positions with the tiled image tensors

This module re-exports that collator so project code can import it from a
stable local path::

    from src.training.collator import concat_pad_data_collator

PYTHONPATH requirement
----------------------
``third_party/InternVL/internvl_chat`` must be on ``PYTHONPATH``.
The launch scripts (``scripts/04_finetune.sh``, ``src/training/train.py``)
handle this automatically.
"""

try:
    from internvl.patch import concat_pad_data_collator  # noqa: F401  — re-export
    _INTERNVL_AVAILABLE = True
except ImportError:
    _INTERNVL_AVAILABLE = False
    concat_pad_data_collator = None  # type: ignore[assignment]

__all__ = ["concat_pad_data_collator"]
