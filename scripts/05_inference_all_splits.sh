#!/usr/bin/bash
#SBATCH --job-name all_inference
#SBATCH --account OPEN-37-27
#SBATCH --nodes 1
#SBATCH --gpus-per-node 8
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Runs inference over all three splits in parallel as a SLURM job array.
#   task 0 → local val   data/drivelm_custom_split/val.jsonl
#   task 1 → local test  data/drivelm_custom_split/test.jsonl
#   task 2 → orig test   data/drivelm/v1_1_val_nus_q_only_converted_llama.json  (+submission.json)
#
# Usage:
#   sbatch scripts/05_inference_all_splits.sh               # submit all 3 splits in parallel
#   sbatch --array=1 scripts/05_inference_all_splits.sh     # re-run just orig test
#   bash   scripts/05_inference_all_splits.sh               # interactive: runs task 0 (override SLURM_ARRAY_TASK_ID below)
#   SLURM_ARRAY_TASK_ID=1 bash scripts/05_inference_all_splits.sh


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── environment ───────────────────────────────────────────────────────────────
source "$ROOT/vqa-ad-ctu-env/bin/activate"

# ── paths ─────────────────────────────────────────────────────────────────────
IMAGES_ROOT=/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched/
OUTPUT_DIR=inference/outputs
export HF_HOME=/mnt/proj1/eu-25-10/dmytro/huggingface_cache

# ── MODEL — only line you need to change between experiments ─────────────────
# MODEL=OpenGVLab/InternVL2-2B
# MODEL=OpenGVLab/Mini-InternVL2-2B-DA-DriveLM
# MODEL=dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct
# MODEL=dkhursen/InternVL2-2b-LoRA-300k-drivelm
# MODEL=dkhursen/InternVL2-2b-LoRA-25k-drivelm
MODEL=llava-hf/llava-v1.6-mistral-7b-hf

IMAGE_MODE=stitched
BATCH_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# InternVL-only tiling settings — left unset for LLaVA/other backends so the
# model can use its own native resolution and tile count.
INTERNVL_ARGS=()
if [[ "$MODEL" == *"InternVL"* || "$MODEL" == *"internvl"* ]]; then
    INTERNVL_ARGS=(--max-tiles 12 --input-size 448)
fi

# ── split table indexed by SLURM_ARRAY_TASK_ID ───────────────────────────────
# format: "ANNOTATIONS_PATH:run_submission(0|1)"
SPLITS=(
    "data/drivelm_custom_split/val.jsonl:0"
    "data/drivelm_custom_split/test.jsonl:0"
    "data/drivelm/v1_1_val_nus_q_only_converted_llama.json:1"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"   # default to 0 when run interactively
ENTRY="${SPLITS[$TASK_ID]}"
ANNOTATIONS="${ENTRY%%:*}"
SUBMISSION="${ENTRY##*:}"

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
SPLIT_STEM=$(basename "$ANNOTATIONS"); SPLIT_STEM="${SPLIT_STEM%.*}"
OUTPUT_FILE="$OUTPUT_DIR/$MODEL_SLUG/${SPLIT_STEM}.json"

if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    RUN_TAG="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
else
    RUN_TAG="interactive_$(date +%Y%m%d_%H%M%S)"
fi
LOG_FILE="inference/logs/${MODEL_SLUG}_${SPLIT_STEM}_${RUN_TAG}.log"

echo "Split      : $ANNOTATIONS"
echo "Output     : $OUTPUT_FILE"
echo "Log        : $LOG_FILE"

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
    "${INTERNVL_ARGS[@]}" \
    --device         cuda \
    2>&1 | tee "$LOG_FILE"

# ── submission wrap (orig test only) ─────────────────────────────────────────
if [[ "$SUBMISSION" -eq 1 ]]; then
    echo "Preparing DriveLM submission from $OUTPUT_FILE ..."
    python -m src.utils.prepare_submission \
        --input_json    "$OUTPUT_FILE" \
        --output_folder "$(dirname "$OUTPUT_FILE")" \
        --method        "$MODEL_SLUG" \
        --output_name   "submission"
fi
