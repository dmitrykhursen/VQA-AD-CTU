"""
Data loading utilities for InternVL2-2B fine-tuning.

The core dataset class lives in InternVL's own codebase:
``third_party/InternVL/internvl_chat/internvl/train/dataset.py``

This module re-exports the key symbols used during training and provides a thin
validation helper to catch missing files before launching a long job.

PYTHONPATH requirement
----------------------
``third_party/InternVL/internvl_chat`` must be on ``PYTHONPATH`` before the
``internvl`` package is importable.  The launch scripts handle this:

    scripts/04_finetune.sh          → sets PYTHONPATH automatically
    src/training/train.py           → sets PYTHONPATH automatically

Or set it manually::

    export PYTHONPATH="$PROJECT_ROOT/third_party/InternVL/internvl_chat:$PROJECT_ROOT"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Re-exports from InternVL's dataset module
# ---------------------------------------------------------------------------

try:
    from internvl.train.dataset import (  # noqa: F401  — re-export
        LazySupervisedDataset,
        WeightedConcatDataset,
        build_transform,
        dynamic_preprocess,
        preprocess,
    )
    _INTERNVL_AVAILABLE = True
except ImportError:
    _INTERNVL_AVAILABLE = False
    LazySupervisedDataset = None       # type: ignore[assignment,misc]
    WeightedConcatDataset = None       # type: ignore[assignment,misc]
    build_transform = None             # type: ignore[assignment]
    dynamic_preprocess = None          # type: ignore[assignment]
    preprocess = None                  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def check_meta(meta_path: Union[str, Path]) -> int:
    """Load and validate an InternVL training meta JSON.

    Verifies that the meta file exists, that every referenced annotation JSONL
    exists, and returns the total number of training samples (accounting for
    ``repeat_time``).

    Parameters
    ----------
    meta_path:
        Path to a JSON file in the format produced by
        ``src/training/build_internvl_meta.py``::

            {
              "dataset_name": {
                "root":        "/abs/path/to/images/",
                "annotation":  "/abs/path/to/train_stitched.jsonl",
                "data_augment": false,
                "repeat_time":  1,
                "length":       25825
              }
            }

    Returns
    -------
    int
        Total number of training samples (sum of ``length × repeat_time``
        across all dataset entries).

    Raises
    ------
    FileNotFoundError
        If the meta file or any annotation JSONL is missing.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with meta_path.open() as f:
        meta = json.load(f)

    total = 0
    for name, cfg in meta.items():
        ann = Path(cfg["annotation"])
        if not ann.exists():
            raise FileNotFoundError(
                f"Annotation file missing for dataset '{name}': {ann}"
            )
        length = cfg.get("length")
        if length is None:
            with ann.open() as fann:
                length = sum(1 for line in fann if line.strip())
        total += length * cfg.get("repeat_time", 1)

    return total
