#!/usr/bin/bash
#SBATCH --job-name inference
#SBATCH --account OPEN-37-27
#SBATCH --nodes 1
#SBATCH --gpus-per-node 8
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Batched inference over the DriveLM-nuScenes test split using InternVL2.
#
# Usage:
#   bash scripts/05_inference.sh                          # interactive / local
#   bash scripts/05_inference.sh --limit 100              # smoke-test on 100 samples
#   bash scripts/05_inference.sh --submission             # also wrap output as submission.json
#   sbatch scripts/05_inference.sh                        # SLURM cluster (single node, N GPUs)
#
# tested max GPU batch-size guide (stitched / separate image mode):
#   H100 64 GB  →  BATCH_SIZE=16 (8 is optimal wrt bandwidth)  / 4? (not tested)

#   A100 40 GB  →  BATCH_SIZE=8  / 1? (not tested)
#
# MODEL can be a short name (see src/models/internvl.py MODEL_PATHS),
# a raw HuggingFace repo id, or a local checkpoint directory path.


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── optional CLI args ─────────────────────────────────────────────────────────
LIMIT_ARG=""
SUBMISSION=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)      LIMIT_ARG="--limit $2"; shift 2 ;;
        --submission) SUBMISSION=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── environment ───────────────────────────────────────────────────────────────
source "$ROOT/vqa-ad-ctu-env/bin/activate"

# ── paths (mainly once set up) ────────────────────────────────────
IMAGES_ROOT=/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched/
# IMAGES_ROOT=/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched_circle_adapt_ctags_bckgrnd/


# ── custom i.i.d. split (local split) ──
ANNOTATIONS=data/drivelm_custom_split/test.jsonl
# ANNOTATIONS=data/drivelm_custom_split/val.jsonl

# ── original DriveLM splits (converted llama format) ──
# ANNOTATIONS=data/drivelm/v1_1_val_nus_q_only_converted_llama.json
# ANNOTATIONS=data/drivelm_custom_split/val.jsonl


OUTPUT_DIR=inference/outputs
export HF_HOME=/mnt/proj1/eu-25-10/dmytro/huggingface_cache   # keeps weights off your home quota

# ── run configuration (change between experiments) ────────────────────────────
# InternVL2 fine-tuned
# MODEL=dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct
# MODEL=dkhursen/InternVL2-2b-LoRA-300k-drivelm
# MODEL=dkhursen/InternVL2-2b-LoRA-25k-drivelm
# MODEL=dkhursen/InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd


# Pretrained baselines
# MODEL=OpenGVLab/InternVL2-2B
# MODEL=llava-hf/llava-v1.6-mistral-7b-hf
MODEL=OpenGVLab/llama_adapter_v2_multimodal7b

IMAGE_MODE=stitched
BATCH_SIZE=8

# InternVL-specific preprocessing (ignored automatically for LLaVA and other backends)
MAX_TILES=12
INPUT_SIZE=448

# ── GPU detection (SLURM or local) ────────────────────────────────────────────
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NUM_GPUS="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    NUM_GPUS=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
echo "Running on $NUM_GPUS GPU(s)"

MODEL_SLUG="${MODEL//\//__}"
SPLIT_STEM=$(basename "$ANNOTATIONS")
SPLIT_STEM="${SPLIT_STEM%.*}"                              # strip .jsonl / .json
OUTPUT_FILE="$OUTPUT_DIR/$MODEL_SLUG/${SPLIT_STEM}.json"  # base path passed to batch_predict

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN_TAG="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"      
else
    RUN_TAG="interactive_$(date +%Y%m%d_%H%M%S)"
fi
LOG_FILE="inference/logs/${MODEL_SLUG}_${RUN_TAG}.log"
echo "Log: $LOG_FILE"

mkdir -p "$OUTPUT_DIR/$MODEL_SLUG" inference/logs

# ── inference ─────────────────────────────────────────────────────────────────
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    -m src.inference.batch_predict \
    --model          "$MODEL" \
    --annotations    "$ANNOTATIONS" \
    --images-root    "$IMAGES_ROOT" \
    --output         "$OUTPUT_FILE" \
    --image-mode     "$IMAGE_MODE" \
    --batch-size     "$BATCH_SIZE" \
    --num-workers    4 \
    --max-new-tokens 512 \
    --max-tiles      "$MAX_TILES" \
    --input-size     "$INPUT_SIZE" \
    --device         cuda \
    ${LIMIT_ARG:+$LIMIT_ARG} \
    2>&1 | tee "$LOG_FILE"

# ── prepare submission ────────────────────────────────────────────────────────
if [[ $SUBMISSION -eq 1 && "$ANNOTATIONS" == *"v1_1_val_nus_q_only"* ]]; then
    # mirror batch_predict: it appends _limit_N to the stem when --limit is used
    SUBMISSION_NAME="submission"
    if [[ -n "$LIMIT_ARG" ]]; then
        LIMIT_VAL="${LIMIT_ARG#--limit }"
        OUTPUT_FILE="${OUTPUT_FILE%.json}_limit_${LIMIT_VAL}.json"
        SUBMISSION_NAME="submission_limit_${LIMIT_VAL}"
    fi
    echo "Preparing DriveLM submission from $OUTPUT_FILE ..."
    python -m src.utils.prepare_submission \
        --input_json    "$OUTPUT_FILE" \
        --output_folder "$(dirname "$OUTPUT_FILE")" \
        --method        "$MODEL_SLUG" \
        --output_name   "$SUBMISSION_NAME"
fi

