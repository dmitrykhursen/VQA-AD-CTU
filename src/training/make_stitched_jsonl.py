"""
Convert a multi-image JSONL (6 camera paths per sample) to a stitched-image JSONL
(single '<frame_id>.jpg' path per sample).

The stitched image filename is derived from the sample id:
  id = "<scene_token>_<frame_token>_<qa_idx>"  ->  image = "<scene_token>_<frame_token>.jpg"

Usage:
    python src/training/make_stitched_jsonl.py \\
        --input  data/drivelm_custom_split/train.jsonl \\
        --output data/drivelm_custom_split/train_stitched.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert multi-camera JSONL to stitched-image JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input JSONL (or JSON array) file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to output JSONL file.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list:
    """Auto-detect JSON array vs. newline-delimited JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"WARNING: skipping malformed line {lineno}: {exc}", file=sys.stderr)
    return records


def derive_frame_key(sample_id: str) -> str:
    """
    '<scene_token>_<frame_token>_<qa_idx>'  ->  '<scene_token>_<frame_token>'
    Falls back to the full id if there is no underscore.
    """
    parts = sample_id.rsplit("_", maxsplit=1)
    return parts[0] if len(parts) == 2 else sample_id


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: input file not found: {args.input}")

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for item in records:
            sample_id = item.get("id", "")
            frame_key = derive_frame_key(sample_id)
            out_item = dict(item)
            out_item["image"] = frame_key + ".jpg"
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Wrote {converted} stitched records to {args.output}")


if __name__ == "__main__":
    main()
