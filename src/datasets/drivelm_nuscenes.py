# Thesis ref: Section 6.1
# PyTorch Dataset for DriveLM-nuScenes JSONL splits.
# Supports stitched (2×3 surround collage) or separate (6-camera multi-image) modes.
#
# Two annotation formats are supported, detected automatically:
#   - JSONL (custom split):  newline-delimited JSON, one record per line
#   - Converted llama format (original DriveLM):  JSON array produced by
#     v1_1_*_converted_llama.json; image paths carry a "data/nuscenes/" prefix
#     that is stripped before joining with images_root

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from src.inference.image_utils import load_pixel_values, surround_collage

# Canonical nuScenes camera order used by DriveLM JSONL image lists.
_CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

# Prefix present on image paths in the converted llama format.
_LLAMA_IMAGE_PREFIX = "data/nuscenes/"


def _load_records(path: Path) -> list[dict]:
    with open(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            # Converted llama format — single JSON array
            return json.load(fh)
        # Custom JSONL split — one JSON object per line
        return [json.loads(line) for line in fh if line.strip()]


def _strip_llama_prefix(p: str) -> str:
    """Remove the 'data/nuscenes/' prefix present in converted llama image paths."""
    return p[len(_LLAMA_IMAGE_PREFIX):] if p.startswith(_LLAMA_IMAGE_PREFIX) else p


class DriveLMNuScenesDataset(Dataset):
    """DriveLM-nuScenes QA dataset for batched InternVL2 inference.

    Accepts both custom JSONL splits and the original DriveLM converted-llama
    JSON format; the format is detected automatically from the file content.

    Args:
        annotations: Path to a JSONL file (custom split) or a converted-llama
            JSON array (e.g. ``data/drivelm/v1_1_val_nus_q_only_converted_llama.json``).
        images_root: Root directory under which the ``CAM_*`` subdirectories live
            (for separate mode) or pre-stitched collages live (for stitched mode).
        image_mode: How to feed camera images to the model.
            ``"stitched"`` — load pre-stitched 2×3 collage as a single image;
            ``"separate"`` — pass each of the 6 cameras independently.
        input_size: InternVL2 tile resolution in pixels (default 448).
        max_tiles: Maximum dynamic tiles per image (default 12).
    """

    def __init__(
        self,
        annotations: str | Path,
        images_root: str | Path,
        image_mode: Literal["stitched", "separate"] = "stitched",
        input_size: int = 448,
        max_tiles: int = 12,
        precompute_tiles: bool = True,
    ) -> None:
        self.images_root = Path(images_root)
        self.image_mode = image_mode
        self.input_size = input_size
        self.max_tiles = max_tiles
        self.precompute_tiles = precompute_tiles

        self.records = _load_records(Path(annotations))

    def __len__(self) -> int:
        return len(self.records)

    def _abs(self, rel: str) -> Path:
        return self.images_root / rel

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Extract question text, stripping any existing <image> placeholders.
        raw_question = rec["conversations"][0]["value"]
        question_text = raw_question.replace("<image>", "").lstrip("\n")
        answer = rec["conversations"][1]["value"] if len(rec["conversations"]) > 1 else ""

        if self.image_mode == "stitched":
            # Pre-stitched collages are stored as {images_root}/{scene_id}.jpg
            # rec["id"] has a trailing QA index (e.g. "abc_def_0") — strip it.
            scene_id = rec["id"].rsplit("_", 1)[0]
            stitched_path = str(self.images_root / f"{scene_id}.jpg")
            image_paths = [stitched_path]
            if self.precompute_tiles:
                pixel_values = load_pixel_values(
                    stitched_path, input_size=self.input_size, max_num=self.max_tiles
                )
                num_patches_list = [pixel_values.shape[0]]
            else:
                pixel_values = None
                num_patches_list = []

        else:  # separate
            paths = [str(self._abs(_strip_llama_prefix(p))) for p in rec["image"]]
            image_paths = paths
            if self.precompute_tiles:
                per_cam = [
                    load_pixel_values(p, input_size=self.input_size, max_num=self.max_tiles)
                    for p in paths
                ]
                pixel_values = torch.cat(per_cam, dim=0)
                num_patches_list = [t.shape[0] for t in per_cam]
            else:
                pixel_values = None
                num_patches_list = []

        return {
            "id": rec["id"],
            "pixel_values": pixel_values,        # (N_tiles, 3, H, W) — InternVL pre-tiled
            "num_patches_list": num_patches_list, # len 1 (stitched) or 6 (separate)
            "image_paths": image_paths,           # raw file paths; backends add <image> tokens
            "question": question_text,            # plain question, no image tokens
            "answer": answer,
            "category": rec.get("category", ""),
            "metric_type": rec.get("metric_type", ""),
        }
