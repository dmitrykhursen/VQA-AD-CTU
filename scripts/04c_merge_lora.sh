#!/usr/bin/bash
# =============================================================================
# Stage 4c – Merge LoRA adapter weights into the base model
#
# Can be run on a login node or compute node (no SLURM header needed).
#
# Usage examples:
#   # Batch-merge every checkpoint-* inside a run directory:
#   bash scripts/04c_merge_lora.sh --all_folder ckpts/finetune/internvl2_2b_lora_25k
#
#   # Single merge:
#   bash scripts/04c_merge_lora.sh \
#       --lora_dir  ckpts/finetune/internvl2_2b_lora_25k/checkpoint-500 \
#       --merged_dir ckpts/finetune/internvl2_2b_lora_25k_merged
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
INTERNVL_DIR="$PROJECT_ROOT/third_party/InternVL/internvl_chat"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load CUDA/12.8 GCC/12.3.0

source "$PROJECT_ROOT/vqa-ad-ctu-env/bin/activate"

export HF_HOME=/mnt/proj1/eu-25-10/dmytro/huggingface_cache
export PYTHONPATH="${PYTHONPATH:-}:${INTERNVL_DIR}:${PROJECT_ROOT}"
export HOME=/mnt/proj1/eu-25-10/dmytro

# ---------------------------------------------------------------------------
# Argument parsing & dispatch
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  Batch merge:  bash $0 --all_folder <run_dir>"
    echo "  Single merge: bash $0 --lora_dir <lora_dir> --merged_dir <output_dir>"
    exit 1
fi

# Forward all arguments verbatim to the Python script
python "$PROJECT_ROOT/src/training/merge_lora.py" "$@"
