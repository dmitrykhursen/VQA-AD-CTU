"""Step 4 of the custom DriveLM split pipeline.

Splits a flat-list DriveLM dataset into train / val / test by scene,
stratified by each scene's dominant QA category, and writes JSONL files.

Usage:
    python src/utils/split_dataset.py \
        --input     data/drivelm_custom_split/intermediate/03_llama.json \
        --output_dir data/drivelm_custom_split \
        --train_ratio 0.8 --val_ratio 0.1 --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


def _primary_category(records: list[dict]) -> str:
    cats = [r.get("category", "unknown") for r in records]
    return Counter(cats).most_common(1)[0][0]


def _stratified_scene_split(
    scene_to_records: dict,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    category_to_scenes: dict[str, list] = defaultdict(list)
    for scene_id, records in scene_to_records.items():
        cat = _primary_category(records)
        category_to_scenes[cat].append(scene_id)

    rng = random.Random(seed)
    train_scenes, val_scenes, test_scenes = [], [], []

    for cat, scenes in sorted(category_to_scenes.items()):
        rng.shuffle(scenes)
        n = len(scenes)
        train_end = max(1, int(n * train_ratio))
        val_end   = train_end + max(1, int(n * val_ratio))
        if n >= 3:
            train_scenes.extend(scenes[:train_end])
            val_scenes.extend(scenes[train_end:val_end])
            test_scenes.extend(scenes[val_end:])
        else:
            train_scenes.extend(scenes)

    return train_scenes, val_scenes, test_scenes


def _report(name: str, records: list[dict]) -> None:
    total = len(records)
    categories = Counter(r.get("category",    "unknown") for r in records)
    metrics    = Counter(r.get("metric_type", "unknown") for r in records)
    print(f"  {name} ({total} records):")
    for cat, count in sorted(categories.items()):
        print(f"    category/{cat}: {count} ({100*count/total:.1f}%)")
    for mt, count in sorted(metrics.items()):
        print(f"    metric/{mt}:    {count} ({100*count/total:.1f}%)")


def _strip_image_prefix(records: list[dict]) -> list[dict]:
    prefix = "data/nuscenes/samples/"
    for rec in records:
        rec["image"] = [
            p[len(prefix):] if p.startswith(prefix) else p
            for p in rec.get("image", [])
        ]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a DriveLM flat-list JSON into train/val/test JSONL by scene."
    )
    parser.add_argument("--input",       required=True,       help="Flat-list JSON from convert2llama.py.")
    parser.add_argument("--output_dir",  required=True,       help="Directory for train.jsonl / val.jsonl / test.jsonl.")
    parser.add_argument("--train_ratio", type=float, default=0.8,  help="Train fraction (default 0.8).")
    parser.add_argument("--val_ratio",   type=float, default=0.1,  help="Val fraction (default 0.1). Test gets the rest.")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed (default 42).")
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for a test set.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.input}")
    with open(args.input) as f:
        raw = json.load(f)

    scene_to_records: dict[str, list] = defaultdict(list)
    skipped = 0
    for rec in raw:
        if "image" not in rec or "id" not in rec:
            skipped += 1
            continue
        scene_id = rec["id"].split("_")[0]
        scene_to_records[scene_id].append(rec)

    scene_ids = list(scene_to_records.keys())
    print(f"Found {len(raw)} records across {len(scene_ids)} scenes ({skipped} skipped).")

    train_scenes, val_scenes, test_scenes = _stratified_scene_split(
        scene_to_records, args.train_ratio, args.val_ratio, args.seed
    )

    def collect(scenes: list) -> list:
        records = []
        for s in scenes:
            records.extend(scene_to_records[s])
        return records

    splits = {
        "train": (train_scenes, collect(train_scenes)),
        "val":   (val_scenes,   collect(val_scenes)),
        "test":  (test_scenes,  collect(test_scenes)),
    }

    print("\nDistribution report (stratified by category):")
    for name, (scenes, records) in splits.items():
        _report(f"{name} ({len(scenes)} scenes)", records)

    print()
    for name, (scenes, records) in splits.items():
        records = _strip_image_prefix(records)
        out_path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {name}: {len(records)} records from {len(scenes)} scenes → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
