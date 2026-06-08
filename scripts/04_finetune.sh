#!/usr/bin/bash
# =============================================================================
# Stage 4 – InternVL2-2B LoRA fine-tuning on DriveLM-25k (train split)
# =============================================================================
#
# Usage (interactive / debug):
#   bash scripts/04_finetune.sh
#
# Usage (SLURM batch):
#   sbatch scripts/04_finetune.sh
#
# =============================================================================
#SBATCH --job-name finetune-internvl2-lora
#SBATCH --account  OPEN-37-27
#SBATCH --nodes    1
#SBATCH --gpus-per-node 8
#SBATCH --partition qgpu
#SBATCH --time     24:00:00
#SBATCH --output   logs/%x_%j.out
#SBATCH --error    logs/%x_%j.err
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
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE=offline
export WANDB_PROJECT=internvl
export HOME=/mnt/proj1/eu-25-10/dmytro
export TRITON_CACHE_DIR="/tmp/triton_${SLURM_JOB_ID:-local}"

MASTER_PORT=$((29000 + RANDOM % 1000))

# ---------------------------------------------------------------------------
# Hardware / batch-size
# ---------------------------------------------------------------------------
GPUS=8
BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))   # = 2

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL="${HF_HOME}/hub/models--OpenGVLab--InternVL2-2B/snapshots/e4f6747bd20f139e637642c6a058c6bd00b36919"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
IMAGES_ROOT="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched/"

# Build stitched JSONLs on-the-fly if they don't exist yet
if [ ! -f "$PROJECT_ROOT/data/drivelm_custom_split/train_stitched.jsonl" ]; then
    echo "[04_finetune] Building train_stitched.jsonl ..."
    python "$PROJECT_ROOT/src/training/make_stitched_jsonl.py" \
        --input  "$PROJECT_ROOT/data/drivelm_custom_split/train.jsonl" \
        --output "$PROJECT_ROOT/data/drivelm_custom_split/train_stitched.jsonl"
fi

if [ ! -f "$PROJECT_ROOT/data/drivelm_custom_split/val_stitched.jsonl" ]; then
    echo "[04_finetune] Building val_stitched.jsonl ..."
    python "$PROJECT_ROOT/src/training/make_stitched_jsonl.py" \
        --input  "$PROJECT_ROOT/data/drivelm_custom_split/val.jsonl" \
        --output "$PROJECT_ROOT/data/drivelm_custom_split/val_stitched.jsonl"
fi

# Build InternVL meta JSONs
TRAIN_META="$PROJECT_ROOT/data/drivelm_custom_split/internvl_meta_train.json"
VAL_META="$PROJECT_ROOT/data/drivelm_custom_split/internvl_meta_val.json"

echo "[04_finetune] Building InternVL train meta ..."
python "$PROJECT_ROOT/src/training/build_internvl_meta.py" \
    --name       drivelm_train \
    --root       "${IMAGES_ROOT}" \
    --annotation "$PROJECT_ROOT/data/drivelm_custom_split/train_stitched.jsonl" \
    --output     "${TRAIN_META}"

echo "[04_finetune] Building InternVL val meta ..."
python "$PROJECT_ROOT/src/training/build_internvl_meta.py" \
    --name              drivelm_val \
    --root              "${IMAGES_ROOT}" \
    --annotation        "$PROJECT_ROOT/data/drivelm_custom_split/val_stitched.jsonl" \
    --output            "${VAL_META}" \
    --max-dynamic-patch 12

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR="$PROJECT_ROOT/ckpts/finetune/internvl2_2b_lora_25k"
mkdir -p "${OUTPUT_DIR}"

RUN_NAME="$(basename "${OUTPUT_DIR}")_${SLURM_JOB_ID:-interactive}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[04_finetune] Starting torchrun – run_name=${RUN_NAME}"
cd "$INTERNVL_DIR"

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    internvl/train/internvl_chat_finetune.py \
    --model_name_or_path          ${MODEL} \
    --conv_style                  "internlm2-chat" \
    --output_dir                  ${OUTPUT_DIR} \
    --overwrite_output_dir        True \
    --freeze_llm                  True \
    --freeze_mlp                  True \
    --freeze_backbone             True \
    --use_llm_lora                16 \
    --meta_path                   ${TRAIN_META} \
    --force_image_size            448 \
    --max_dynamic_patch           12 \
    --down_sample_ratio           0.5 \
    --pad2square                  False \
    --dynamic_image_size          True \
    --use_thumbnail               True \
    --ps_version                  'v2' \
    --vision_select_layer         -1 \
    --use_data_resampling         False \
    --max_seq_length              8192 \
    --dataloader_num_workers      4 \
    --do_train                    True \
    --num_train_epochs            10 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --learning_rate               4e-5 \
    --weight_decay                0.01 \
    --warmup_ratio                0.03 \
    --lr_scheduler_type           "cosine" \
    --bf16                        True \
    --grad_checkpoint             True \
    --drop_path_rate              0.0 \
    --eval_meta_path              ${VAL_META} \
    --evaluation_strategy         "epoch" \
    --per_device_eval_batch_size  4 \
    --inference_batch_size        8 \
    --save_strategy               "epoch" \
    --save_total_limit            10 \
    --logging_steps               10 \
    --report_to                   "wandb" \
    --run_name                    "${RUN_NAME}" \
    --loss_reduction              "token" \
    --remove_unused_columns       False \
    --group_by_length             False \
    --seed                        42 \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "[04_finetune] Done."
