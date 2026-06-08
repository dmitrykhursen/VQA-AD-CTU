#!/usr/bin/env bash
#SBATCH --job-name qa_gen_nuscenes
#SBATCH --account TODO_YOUR_PROJECT_ID
#SBATCH --nodes 1
#SBATCH --gpus-per-node 8
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --output=logs/03_qa_gen_%x_%j.out
#SBATCH --error=logs/03_qa_gen_%x_%j.err
#
# Generate QA pairs from NuScenes annotations + tracks via an LLM.
#
# Processes one scene × one camera at a time.  Override via env vars:
#   SCENE=n008-2018-05-21-11-06-59-0400
#   CAMERA=CAM_FRONT
#   MODEL=Qwen/Qwen3-14B
#   bash scripts/03_run_pipeline_nuscenes.sh
#
# For a full run, wrap this script in a loop over scenes and cameras,
# or submit as a SLURM array with different SCENE/CAMERA values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT/vqa-ad-ctu-env/bin/activate"

module load CUDA/12.6 2>/dev/null || true

# ── scene / camera selection ──────────────────────────────────────────────
SCENE="${SCENE:-n008-2018-05-21-11-06-59-0400}"
CAMERA="${CAMERA:-CAM_FRONT}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"

# ── local data paths (produced by step 02) ────────────────────────────────
METADATA_DIR="$ROOT/data/nuscenes-drivelm_metadata"
YOLO_PATH="$METADATA_DIR/object_annotations/$CAMERA/$SCENE"
TRACKS_PATH="$METADATA_DIR/object_tracks/$SCENE/tracks.json"

# ── config paths ──────────────────────────────────────────────────────────
PROMPTS_CONFIG="$ROOT/configs/pipeline/llm_prompt_config.yaml"
QAS_RATIOS="$ROOT/configs/pipeline/drivelm_qas_ratios_to_gen.json"
OUTPUT_DIR="$ROOT/data/drivelm_aug_pseudo_labels"
OUTPUT_FILE="$OUTPUT_DIR/drivelm_pseudo_qas.json"

echo "=== NuScenes QA generation ==="
echo "Scene   : $SCENE"
echo "Camera  : $CAMERA"
echo "Model   : $MODEL"
echo "Output  : $OUTPUT_FILE"
echo

mkdir -p "$OUTPUT_DIR" "$ROOT/logs"

# Note: qa_generation.py appends to a .jsonl per chunk; the final merged
# JSON is expected at OUTPUT_FILE. Rename/merge as needed after the run.

python3 "$ROOT/src/pipeline/llm_orchestration/qa_generation.py" \
    --model            "$MODEL" \
    --yolo_path        "$YOLO_PATH" \
    --tracks_path      "$TRACKS_PATH" \
    --prompts_config   "$PROMPTS_CONFIG" \
    --qas_ratios       "$QAS_RATIOS" \
    --output_folder    "$OUTPUT_DIR" \
    --file_name        "drivelm_pseudo_qas" \
    --use_tracks \
    --thinking \
    --answer_formatting

echo
echo "=== Done. QAs written to $OUTPUT_FILE ==="
