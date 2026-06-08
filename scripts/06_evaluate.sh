#!/bin/bash
# Run DriveLM evaluation against the custom test split.
#
# Usage:
#   source /path/to/venv/bin/activate
#   bash scripts/06_evaluate.sh <predictions.json> [gt_test.json]
#
# Arguments:
#   predictions.json  — model output file (list of {id, answer, ...})
#   test.jsonl     — ground-truth test split in llama format (default: data/drivelm_custom_split/test.jsonl)
#
# Environment:
#   OPENAI_API_KEY    — required for the ChatGPT metric
#
# Example:
#   export OPENAI_API_KEY="sk-..."
#   bash scripts/06_evaluate.sh inference/outputs/MODEL/local_test.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Java is required by the language-evaluation package (METEOR scorer loads a JVM on import).
# On HPC clusters load it via the module system before running this script, e.g.:
ml Java
# On a standard Linux machine: sudo apt-get install default-jre

source "$ROOT/vqa-ad-ctu-env/bin/activate"

ROOT_PATH1="${1:?Usage: $0 <predictions.json|model_dir> [gt_test.jsonl]}"
ROOT_PATH2="${2:-data/drivelm_custom_split/test.jsonl}"

# Accept a directory: auto-append local_test.json
if [ -d "$ROOT_PATH1" ]; then
    ROOT_PATH1="${ROOT_PATH1%/}/local_test.json"
fi

# Derive model name from the parent directory of the prediction file
# e.g. inference/outputs/OpenGVLab__InternVL2-2B/local_test.json -> OpenGVLab__InternVL2-2B
MODEL_NAME="$(basename "$(dirname "$ROOT_PATH1")")"
OUTPUT_DIR="evaluation/results/${MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before running evaluation."
    exit 1
fi

echo "Model:      $MODEL_NAME"
echo "Preds:      $ROOT_PATH1"
echo "GT:         $ROOT_PATH2"
echo "Output dir: $OUTPUT_DIR"

PYTHONPATH="." python src/evaluation/evaluation_extended.py \
    --root_path1="$ROOT_PATH1" \
    --root_path2="$ROOT_PATH2" \
    --llama_format \
    --output_dir="$OUTPUT_DIR" \
    --model_name="$MODEL_NAME" \
    --eval_all
