"""
Build the InternVL training-meta JSON that maps dataset names to their
image root and annotation JSONL paths.

The meta format expected by internvl_chat_finetune.py:
    {
      "dataset_name": {
        "root":             "/abs/path/to/stitched/images/",
        "annotation":       "/abs/path/to/train_stitched.jsonl",
        "data_augment":     false,
        "repeat_time":      1,
        "length":           <num_samples>,
        "max_dynamic_patch": 12          # optional, val only
      }
    }

Usage (train meta):
    python src/training/build_internvl_meta.py \\
        --name drivelm_train \\
        --root /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched/ \\
        --annotation data/drivelm_custom_split/train_stitched.jsonl \\
        --output data/drivelm_custom_split/internvl_meta_train.json

Usage (val meta with max_dynamic_patch):
    python src/training/build_internvl_meta.py \\
        --name drivelm_val \\
        --root /mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched/ \\
        --annotation data/drivelm_custom_split/val_stitched.jsonl \\
        --output data/drivelm_custom_split/internvl_meta_val.json \\
        --max-dynamic-patch 12
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an InternVL training-meta JSON file."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name key used inside the meta JSON (e.g. 'drivelm_train').",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Absolute path to the directory containing stitched images. "
             "Will be stored as-is; a trailing slash is recommended.",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        type=Path,
        help="Path to the annotation JSONL file. "
             "Line count is used as the 'length' field.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the output meta JSON file.",
    )
    parser.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=None,
        dest="max_dynamic_patch",
        help="If set, adds 'max_dynamic_patch' to the entry (recommended for val).",
    )
    parser.add_argument(
        "--data-augment",
        action="store_true",
        default=False,
        dest="data_augment",
        help="Set data_augment=true in the meta entry (default: false).",
    )
    parser.add_argument(
        "--repeat-time",
        type=int,
        default=1,
        dest="repeat_time",
        help="Number of times to repeat the dataset per epoch (default: 1).",
    )
    return parser.parse_args()


def count_lines(path: Path) -> int:
    """Count non-empty lines (each line = one sample in JSONL)."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def ensure_trailing_slash(root: str) -> str:
    return root if root.endswith("/") else root + "/"


def main() -> None:
    args = parse_args()

    if not args.annotation.exists():
        sys.exit(f"ERROR: annotation file not found: {args.annotation}")

    length = count_lines(args.annotation)
    root = ensure_trailing_slash(args.root)
    annotation_abs = str(args.annotation.resolve())

    entry: dict = {
        "root": root,
        "annotation": annotation_abs,
        "data_augment": args.data_augment,
        "repeat_time": args.repeat_time,
        "length": length,
    }
    if args.max_dynamic_patch is not None:
        entry["max_dynamic_patch"] = args.max_dynamic_patch

    meta = {args.name: entry}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        json.dump(meta, fout, indent=2)
        fout.write("\n")

    print(
        f"Wrote InternVL meta to {args.output}\n"
        f"  name={args.name!r}  length={length}  root={root!r}"
        + (f"  max_dynamic_patch={args.max_dynamic_patch}" if args.max_dynamic_patch else "")
    )


if __name__ == "__main__":
    main()
