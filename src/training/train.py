"""
Python launcher for InternVL2-2B LoRA fine-tuning.

Delegates to::

    third_party/InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py

via ``torchrun``, using paths derived from this file's location so the command
works from any working directory.

All default hyperparameters match the thesis configuration (Section 6.1.3);
see also ``configs/finetune/internvl2_2b_lora.yaml`` and
:mod:`src.training.lora_config`.

Usage
-----
From the project root::

    python src/training/train.py \\
        --meta-path  data/drivelm_custom_split/internvl_meta_train.json \\
        --eval-meta  data/drivelm_custom_split/internvl_meta_val.json \\
        --output-dir ckpts/finetune/internvl2_2b_lora_25k

Augmented run (DL-PL 10%)::

    python src/training/train.py \\
        --meta-path  data/drivelm_aug_pseudo_labels/internvl_meta_train_aug10pct.json \\
        --eval-meta  data/drivelm_custom_split/internvl_meta_val.json \\
        --output-dir ckpts/finetune/internvl2_2b_lora_25k_dlpl10pct

For SLURM submissions, prefer the wrapper scripts which handle module loads
and environment activation::

    sbatch scripts/04_finetune.sh
    SPLIT_PCT=10 sbatch scripts/04b_finetune_aug.sh
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from src.training.data_loader import check_meta
from src.training.lora_config import DEFAULT_LORA

# ---------------------------------------------------------------------------
# Project-relative paths (resolved from this file, independent of cwd)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
INTERNVL_CHAT_DIR = PROJECT_ROOT / "third_party" / "InternVL" / "internvl_chat"
FINETUNE_SCRIPT = INTERNVL_CHAT_DIR / "internvl" / "train" / "internvl_chat_finetune.py"

_DEFAULT_MODEL = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    / "hub/models--OpenGVLab--InternVL2-2B/snapshots"
    / "e4f6747bd20f139e637642c6a058c6bd00b36919"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch InternVL2-2B LoRA fine-tuning via torchrun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    p.add_argument(
        "--meta-path", required=True, type=Path,
        help="InternVL training meta JSON (built by src/training/build_internvl_meta.py).",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory for checkpoints and training logs.",
    )

    # --- Optional data ---
    p.add_argument(
        "--eval-meta", default=None, type=Path,
        help="InternVL val meta JSON.  When set, runs DriveLM metrics after each epoch.",
    )
    p.add_argument(
        "--model", default=str(_DEFAULT_MODEL), type=str,
        help="Path to the base InternVL2-2B HuggingFace checkpoint.",
    )

    # --- Hardware ---
    p.add_argument("--gpus", default=8, type=int, help="Number of GPUs (nproc_per_node).")
    p.add_argument("--batch-size", default=64, type=int, help="Effective global batch size.")
    p.add_argument("--per-device-batch-size", default=4, type=int)

    # --- Training schedule ---
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--lr", default=4e-5, type=float)
    p.add_argument("--lora-rank", default=DEFAULT_LORA.rank, type=int,
                   help="LoRA rank r (alpha = 2r is set by InternVL internals).")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--max-seq-length", default=8192, type=int)

    # --- Misc ---
    p.add_argument("--run-name", default=None, type=str,
                   help="W&B run name (defaults to output-dir basename).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the torchrun command without executing it.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_cmd(args: argparse.Namespace) -> list[str]:
    """Construct the full ``torchrun`` command as a list of strings."""

    grad_acc = max(1, args.batch_size // args.per_device_batch_size // args.gpus)
    run_name = args.run_name or args.output_dir.name

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        f"--nproc_per_node={args.gpus}",
        f"--master_port={_free_port()}",
        str(FINETUNE_SCRIPT),

        # model
        "--model_name_or_path",          args.model,
        "--conv_style",                  "internlm2-chat",
        "--output_dir",                  str(args.output_dir),
        "--overwrite_output_dir",        "True",

        # LoRA (alpha = 2*rank is set by InternVL internals)
        "--freeze_llm",                  "True",
        "--freeze_mlp",                  "True",
        "--freeze_backbone",             "True",
        "--use_llm_lora",                str(args.lora_rank),

        # data / image preprocessing
        "--meta_path",                   str(args.meta_path),
        "--force_image_size",            "448",
        "--max_dynamic_patch",           "12",
        "--down_sample_ratio",           "0.5",
        "--pad2square",                  "False",
        "--dynamic_image_size",          "True",
        "--use_thumbnail",               "True",
        "--ps_version",                  "v2",
        "--vision_select_layer",         "-1",
        "--use_data_resampling",         "False",
        "--max_seq_length",              str(args.max_seq_length),
        "--dataloader_num_workers",      "4",

        # training schedule
        "--do_train",                    "True",
        "--num_train_epochs",            str(args.epochs),
        "--per_device_train_batch_size", str(args.per_device_batch_size),
        "--gradient_accumulation_steps", str(grad_acc),
        "--learning_rate",               str(args.lr),
        "--weight_decay",                "0.01",
        "--warmup_ratio",                "0.03",
        "--lr_scheduler_type",           "cosine",
        "--bf16",                        "True",
        "--grad_checkpoint",             "True",
        "--drop_path_rate",              "0.0",

        # checkpointing
        "--save_strategy",               "epoch",
        "--save_total_limit",            "10",

        # logging
        "--logging_steps",               "10",
        "--report_to",                   "wandb",
        "--run_name",                    run_name,

        # loss / misc
        "--loss_reduction",              "token",
        "--remove_unused_columns",       "False",
        "--group_by_length",             "False",
        "--seed",                        str(args.seed),
    ]

    if args.eval_meta is not None:
        cmd += [
            "--eval_meta_path",              str(args.eval_meta),
            "--evaluation_strategy",         "epoch",
            "--per_device_eval_batch_size",  "4",
            "--inference_batch_size",        "8",
        ]

    return cmd


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate paths
    if not FINETUNE_SCRIPT.exists():
        sys.exit(
            f"ERROR: InternVL finetune script not found at {FINETUNE_SCRIPT}\n"
            f"  Clone and install InternVL first:\n"
            f"    git clone https://github.com/OpenGVLab/InternVL third_party/InternVL\n"
            f"    cd third_party/InternVL/internvl_chat && pip install -e '[train]'\n"
            f"  See third_party/InternVL/README.md for details."
        )

    check_meta(args.meta_path)
    if args.eval_meta:
        check_meta(args.eval_meta)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Add InternVL to PYTHONPATH so the finetune script can import internvl.*
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{INTERNVL_CHAT_DIR}:{PROJECT_ROOT}:{existing}" if existing
        else f"{INTERNVL_CHAT_DIR}:{PROJECT_ROOT}"
    )

    cmd = build_cmd(args)

    if args.dry_run:
        print("torchrun command (dry run):")
        print("  " + " \\\n  ".join(cmd))
        return

    log_file = args.output_dir / "training_log.txt"
    print(f"[train] Launching torchrun — log: {log_file}")
    print(f"[train] Effective batch: {args.gpus} GPU × {args.per_device_batch_size} × "
          f"{max(1, args.batch_size // args.per_device_batch_size // args.gpus)} grad_acc "
          f"= {args.batch_size}")

    # Run from the internvl_chat directory so relative imports inside the
    # InternVL codebase (e.g. shell/*.sh zero-stage configs) resolve correctly.
    with open(log_file, "a") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(INTERNVL_CHAT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in (proc.stdout or "").splitlines(keepends=True):
            sys.stdout.write(line)
            log.write(line)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
