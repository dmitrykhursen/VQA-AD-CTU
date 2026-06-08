

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.drivelm_nuscenes import DriveLMNuScenesDataset


def _get_backend(model: str):
    """Return the correct model backend module, imported lazily."""
    m = model.lower()
    if "llava" in m:
        import src.models.llava1_6 as backend
    elif "llama_adapter" in m:
        import src.models.llama_adapterv2 as backend
    else:
        import src.models.internvl as backend
    return backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _set_rank_in_log_format(rank: int) -> None:
    fmt = f"%(asctime)s  [GPU:{rank}]  %(levelname)-8s  %(message)s"
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _init_distributed() -> tuple[int, int, int]:
    """Return (rank, local_rank, world_size). Initialises NCCL when world_size > 1."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


# ---------------------------------------------------------------------------
# DataLoader collation
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    """Concatenate variable-tile pixel_values and flatten num_patches_list.

    pixel_values is None when precompute_tiles=False (non-InternVL backends).
    """
    pv_list = [x["pixel_values"] for x in batch]
    return {
        "ids":                    [x["id"] for x in batch],
        "pixel_values":           torch.cat(pv_list, dim=0) if pv_list[0] is not None else None,
        # Flat list: tile count per <image> token, across all samples in the batch.
        # stitched → 1 entry per sample; separate → 6 entries per sample.
        "num_patches_list":       [n for x in batch for n in x["num_patches_list"]],
        "questions":              [x["question"] for x in batch],
        "image_paths_per_sample": [x["image_paths"] for x in batch],
        "answers":                [x["answer"] for x in batch],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batched DriveLM inference — supports InternVL2/2.5/3 and LLaVA-NeXT 1.6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── common args (all backends) ────────────────────────────────────────────
    p.add_argument("--model", required=True,
                   help="Short name (see src/models/internvl.py MODEL_PATHS), "
                        "HF repo id, or local checkpoint path")
    p.add_argument("--annotations", "--split", dest="annotations", required=True,
                   help="Path to JSONL or converted-llama JSON annotation file")
    p.add_argument("--images-root", required=True,
                   help="Root directory for images (CAM_* subdirs or stitched collages)")
    p.add_argument("--output", required=True,
                   help="Output JSON file path")
    p.add_argument("--image-mode", choices=["stitched", "separate"], default="stitched",
                   help="'stitched': pre-stitched 2×3 collage; 'separate': 6 cameras")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Samples per forward pass")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda",
                   help="PyTorch device; overridden per-rank when using torchrun")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate dataset to N samples (quick smoke-test)")

    # ── InternVL-specific args (ignored by other backends) ────────────────────
    iv = p.add_argument_group("InternVL options (ignored for LLaVA and other backends)")
    iv.add_argument("--max-tiles", type=int, default=12,
                    help="Maximum dynamic tiles per image for InternVL tiling")
    iv.add_argument("--input-size", type=int, default=448,
                    help="Tile resolution in pixels for InternVL preprocessing")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rank, local_rank, world_size = _init_distributed()
    _set_rank_in_log_format(rank)
    args = _parse_args()

    # When launched with torchrun each rank owns its own GPU.
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else args.device

    # TF32 is default on Ampere/Hopper but set explicitly for clarity.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        log.info("World size: %d", world_size)
        log.info("Run parameters:")
        for key, val in sorted(vars(args).items()):
            log.info("  %-20s %s", key, val)
        log.info("Loading model: %s", args.model)
    backend = _get_backend(args.model)
    model, tokenizer = backend.load(args.model, device=device)

    dataset = DriveLMNuScenesDataset(
        annotations=args.annotations,
        images_root=args.images_root,
        image_mode=args.image_mode,
        input_size=args.input_size,
        max_tiles=args.max_tiles,
        precompute_tiles=getattr(backend, "NEEDS_TILED_PIXELS", True),
    )

    if args.limit is not None:
        dataset.records = dataset.records[: args.limit]
        if rank == 0:
            log.info("--limit: truncated to %d samples", len(dataset))

    # Slice dataset so each rank owns a disjoint shard — no padding, no duplicates.
    # rank::world_size gives every sample to exactly one GPU, last batch included.
    if world_size > 1:
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))

    if rank == 0:
        log.info(
            "Dataset: %d total samples | per-gpu ~%d | image_mode=%s | batch_size=%d | gpus=%d",
            len(dataset) * world_size, len(dataset), args.image_mode, args.batch_size, world_size,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if args.num_workers > 0 else None,
        
    )

    local_results: list[dict] = []

    pbar = tqdm(loader, desc=f"[rank {rank}] {args.annotations}", unit="batch",
                dynamic_ncols=True, disable=(rank != 0))
    with torch.inference_mode():
        for batch in pbar:
            responses: list[str] = backend.batch_predict(
                (model, tokenizer),
                batch["pixel_values"],
                batch["questions"],
                batch["image_paths_per_sample"],
                num_patches_list=batch["num_patches_list"],
                max_new_tokens=args.max_new_tokens,
                device=device,
            )

            for rec_id, question, answer, prediction in zip(
                batch["ids"], batch["questions"], batch["answers"], responses,
            ):
                local_results.append({
                    "id":           rec_id,
                    "question":     question,
                    "ground_truth": answer,
                    "prediction":   prediction,
                })

        if rank == 0:
            pbar.set_postfix(samples=len(local_results))

    # ── gather results from all ranks to rank 0 ───────────────────────────────
    # Free model weights before the NCCL gather; gather_object internally
    # allocates CUDA tensors to exchange object sizes, which triggers OOM when
    # GPU memory is exhausted after a long inference run.
    del model
    torch.cuda.empty_cache()

    results: list[dict] = []
    if world_size > 1:
        gathered: list[list[dict]] = [[] for _ in range(world_size)]
        dist.gather_object(local_results, gathered if rank == 0 else None, dst=0)
        if rank == 0:
            results = [item for shard in gathered for item in shard]
        dist.destroy_process_group()
    else:
        results = local_results

    # ── write output (rank 0 only) ────────────────────────────────────────────
    if rank == 0:
        out_path = Path(args.output)
        if args.limit is not None:
            out_path = out_path.with_stem(f"{out_path.stem}_limit_{args.limit}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {"args": vars(args), "predictions": results}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=4, ensure_ascii=False)
        log.info("Saved %d predictions → %s", len(results), out_path)


if __name__ == "__main__":
    main()
