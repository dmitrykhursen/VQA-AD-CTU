"""
Merge LoRA adapter weights into the base model.

Modes:
  Single merge:   python src/training/merge_lora.py <lora_dir> <output_dir>
  Batch (folder): python src/training/merge_lora.py --all_folder <work_dir>
                  Finds all checkpoint-* subdirs, merges each to
                  <work_dir>_merged/<checkpoint_name>/

Requires PYTHONPATH to include third_party/InternVL/internvl_chat so that
the internvl package can be imported.
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge_lora(lora_dir: Path, merged_dir: Path) -> None:
    """Load a LoRA checkpoint and save the merged (full-weight) model."""
    lora_dir = lora_dir.resolve()
    merged_dir = merged_dir.resolve()

    print(f"[merge_lora] Loading LoRA checkpoint: {lora_dir}")
    print(f"[merge_lora] Output directory:        {merged_dir}")

    # Load LoRA checkpoint (adapters stored inside the checkpoint directory)
    model = InternVLChatModel.from_pretrained(
        str(lora_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()

    # Merge adapters and strip the PEFT wrapper — must be done separately for
    # vision backbone and LLM because each wraps its own PeftModel.
    if model.config.use_backbone_lora:
        model.vision_model.merge_and_unload()
        model.vision_model = model.vision_model.model
        model.config.use_backbone_lora = 0
    if model.config.use_llm_lora:
        model.language_model.merge_and_unload()
        model.language_model = model.language_model.model
        model.config.use_llm_lora = 0

    tokenizer = AutoTokenizer.from_pretrained(
        str(lora_dir), trust_remote_code=True
    )

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print(f"[merge_lora] Saved merged model to {merged_dir}")


def merge_checkpoints_in_folder(work_dir: Path) -> None:
    """
    Batch mode: find all checkpoint-* subdirectories inside *work_dir* and
    merge each one into <work_dir>_merged/<checkpoint_name>/.
    """
    work_dir = work_dir.resolve()
    if not work_dir.is_dir():
        sys.exit(f"ERROR: --all_folder directory does not exist: {work_dir}")

    # Collect checkpoint directories and sort by step number
    checkpoint_dirs = sorted(
        [d for d in work_dir.iterdir() if d.is_dir() and re.match(r"checkpoint-\d+", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )

    if not checkpoint_dirs:
        sys.exit(f"ERROR: no checkpoint-* subdirectories found in {work_dir}")

    merged_root = work_dir.parent / (work_dir.name + "_merged")
    print(
        f"[merge_lora] Batch mode: {len(checkpoint_dirs)} checkpoint(s) found.\n"
        f"             Output root: {merged_root}"
    )

    for ckpt_dir in checkpoint_dirs:
        out_dir = merged_root / ckpt_dir.name
        merge_lora(ckpt_dir, out_dir)

    print(f"[merge_lora] Batch merge complete. All checkpoints merged to {merged_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge InternVL LoRA adapter weights into the base model."
    )

    # Batch mode
    parser.add_argument(
        "--all_folder",
        type=Path,
        default=None,
        metavar="WORK_DIR",
        help=(
            "Batch mode: merge every checkpoint-* subdir inside WORK_DIR. "
            "Results are saved to <WORK_DIR>_merged/<checkpoint_name>/."
        ),
    )

    # Single-merge mode (positional for backward compat with original script)
    parser.add_argument(
        "--lora_dir",
        type=Path,
        default=None,
        metavar="LORA_DIR",
        help="Path to a single LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--merged_dir",
        type=Path,
        default=None,
        metavar="MERGED_DIR",
        help="Output directory for the merged model.",
    )

    # Positional fallback: <lora_dir> <output_dir>
    parser.add_argument("positional", nargs="*", help=argparse.SUPPRESS)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_folder is not None:
        merge_checkpoints_in_folder(args.all_folder)
        return

    # Resolve lora_dir / merged_dir – accept both named flags and positional args
    lora_dir = args.lora_dir
    merged_dir = args.merged_dir

    if lora_dir is None and len(args.positional) >= 1:
        lora_dir = Path(args.positional[0])
    if merged_dir is None and len(args.positional) >= 2:
        merged_dir = Path(args.positional[1])

    if lora_dir is None or merged_dir is None:
        sys.exit(
            "ERROR: provide either --all_folder <dir>  or  "
            "--lora_dir <dir> --merged_dir <dir>  (or two positional paths)."
        )

    merge_lora(lora_dir, merged_dir)


if __name__ == "__main__":
    main()


# =============================================================================
# Usage notes
# =============================================================================
# Set PYTHONPATH first:
#   export PYTHONPATH="$PROJECT_ROOT/third_party/InternVL/internvl_chat:$PROJECT_ROOT"
#   source vqa-ad-ctu-env/bin/activate
#
# Single merge:
#   python src/training/merge_lora.py \
#       ckpts/finetune/internvl2_2b_lora_25k/checkpoint-500 \
#       ckpts/finetune/internvl2_2b_lora_25k_merged
#
# Batch merge all checkpoints:
#   python src/training/merge_lora.py \
#       --all_folder ckpts/finetune/internvl2_2b_lora_25k
