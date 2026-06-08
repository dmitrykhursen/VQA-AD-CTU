#!/usr/bin/env bash
# Build the DriveLM custom 80/10/10 train/val/test split.
#
# Expected output:
#   data/drivelm_custom_split/train.jsonl
#   data/drivelm_custom_split/val.jsonl
#   data/drivelm_custom_split/test.jsonl
#
# Override the source JSON via the DRIVELM_JSON env var:
#   DRIVELM_JSON=/your/path/v1_1_train_nus_with_all_metainfo.json bash scripts/01_build_custom_split.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT/vqa-ad-ctu-env/bin/activate"

# ── prerequisite: locate source JSON ─────────────────────────────────────
# Preferred: enriched file that includes per-scene scene_description metainfo.
# Fallback:  original train JSON without scene_description.
PREFERRED="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_1_train_nus_with_all_metainf.json"
FALLBACK="/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/v1_1_train_nus.json"

if [ -n "${DRIVELM_JSON:-}" ]; then
    # user-supplied path
    if [ ! -f "$DRIVELM_JSON" ]; then
        echo "ERROR: DRIVELM_JSON='$DRIVELM_JSON' does not exist." >&2
        exit 1
    fi
elif [ -f "$PREFERRED" ]; then
    DRIVELM_JSON="$PREFERRED"
elif [ -f "$FALLBACK" ]; then
    echo "WARNING: preferred source not found; falling back to original train JSON (no scene_description metainfo)."
    DRIVELM_JSON="$FALLBACK"
else
    echo "ERROR: no DriveLM source JSON found. Provide one via DRIVELM_JSON env var." >&2
    exit 1
fi

WORK_DIR="$ROOT/data/drivelm_custom_split"
INTER="$WORK_DIR/intermediate"

echo "=== DriveLM custom split pipeline ==="
echo "Source : $DRIVELM_JSON"
echo "Output : $WORK_DIR"
echo

mkdir -p "$INTER"

# ── Step 1: extract QAs following the test question distribution ──────────
echo "[1/4] Extracting QAs with evaluation tags..."
python3 "$ROOT/src/datasets/extract_data.py" \
    --input  "$DRIVELM_JSON" \
    --output "$INTER/01_extracted.json"

# ── Step 2: augment multiple-choice questions ─────────────────────────────
echo "[2/4] Adding multiple-choice options to MC questions..."
python3 "$ROOT/src/datasets/convert_data.py" \
    --input  "$INTER/01_extracted.json" \
    --output "$INTER/02_converted.json" \
    --seed 0

# ── Step 3: flatten nested structure to LLaMA conversation format ─────────
echo "[3/4] Flattening to conversation format..."
python3 "$ROOT/src/datasets/convert2llama.py" \
    --input  "$INTER/02_converted.json" \
    --output "$INTER/03_llama.json"

# ── Step 4: stratified scene split 80/10/10 → JSONL ──────────────────────
echo "[4/4] Splitting into train / val / test (80/10/10)..."
python3 "$ROOT/src/utils/split_dataset.py" \
    --input      "$INTER/03_llama.json" \
    --output_dir "$WORK_DIR" \
    --train_ratio 0.8 \
    --val_ratio   0.1 \
    --seed 42

echo
echo "=== Done. Splits written to $WORK_DIR ==="
ls -lh "$WORK_DIR"/*.jsonl
