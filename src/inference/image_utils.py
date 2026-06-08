# Thesis ref: Section 6.2
# Image preprocessing for InternVL2 inference: ImageNet-normalized 448x448 transform,
# adaptive tiling (dynamic_preprocess), and 2x3 surround-view collage for nuScenes 6-camera frames.

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CAMERA_GRID = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    cols = target_width // image_size
    processed_images = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _identify_camera(path: str) -> str:
    for key in (
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
    ):
        if key in path:
            return key
    return "UNKNOWN"


def surround_collage(
    images: Sequence[str | Path | Image.Image],
    tile_size: int = 336,
) -> Image.Image:
    """Stitch 6 nuScenes cameras into a 2x3 grid.

    Accepts either a list of file paths (camera identified by substring) or PIL.Images
    in the canonical order [FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT].
    Missing cameras are rendered as black tiles.
    """
    image_map: dict[str, Image.Image] = {}
    if all(isinstance(x, (str, Path)) for x in images):
        for p in images:
            cam = _identify_camera(str(p))
            if cam != "UNKNOWN":
                image_map[cam] = Image.open(p).convert("RGB")
    else:
        canonical = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        for cam, img in zip(canonical, images):
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            image_map[cam] = img

    row_h = tile_size
    row_w = tile_size * 3
    collage = Image.new("RGB", (row_w, row_h * 2), color=(0, 0, 0))
    for r, row in enumerate(CAMERA_GRID):
        for c, cam in enumerate(row):
            if cam in image_map:
                tile = image_map[cam].resize((tile_size, tile_size), Image.BICUBIC)
                collage.paste(tile, (c * tile_size, r * tile_size))
    return collage


def load_pixel_values(
    image: str | Path | Image.Image,
    input_size: int = 448,
    max_num: int = 6,
) -> torch.Tensor:
    """Apply dynamic tiling + ImageNet transform and stack tiles into (N, 3, H, W)."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(t) for t in tiles])
