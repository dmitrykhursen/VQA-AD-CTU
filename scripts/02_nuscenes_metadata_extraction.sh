#!/usr/bin/env bash
# Extract per-image 2D+3D annotations from NuScenes, then aggregate them
# into per-scene object tracks.
#
# Intermediate output:
#   data/nuscenes-drivelm_metadata/object_annotations/<sensor>/<log_id>/<stem>.json
#
# Final output:
#   data/nuscenes-drivelm_metadata/object_tracks/<scene_name>/tracks.json
#
# Override defaults via env vars:
#   NUSCENES_ROOT=/your/path   bash scripts/02_nuscenes_metadata_extraction.sh
#   NUSCENES_VERSION=v1.0-mini bash scripts/02_nuscenes_metadata_extraction.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT/vqa-ad-ctu-env/bin/activate"

# ── dataset location ──────────────────────────────────────────────────────
NUSCENES_ROOT="${NUSCENES_ROOT:-/scratch/project/eu-25-10/datasets/nuScenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
# NUSCENES_VERSION=v1.0-mini bash scripts/02_nuscenes_metadata_extraction.sh

NUM_WORKERS="${NUM_WORKERS:-4}"

if [ ! -d "$NUSCENES_ROOT" ]; then
    echo "ERROR: NuScenes root not found: $NUSCENES_ROOT" >&2
    echo "       Set NUSCENES_ROOT env var to override." >&2
    exit 1
fi



# ── output paths ──────────────────────────────────────────────────────────
METADATA_DIR="$ROOT/data/nuscenes-drivelm_metadata"
ANNOTATIONS_DIR="$METADATA_DIR/object_annotations"  
TRACKS_DIR="$METADATA_DIR/object_tracks"             

echo "=== NuScenes metadata extraction ==="
echo "Source   : $NUSCENES_ROOT  ($NUSCENES_VERSION)"
echo "Metadata : $METADATA_DIR"
echo

mkdir -p "$ANNOTATIONS_DIR" "$TRACKS_DIR"

# ── Step 1: per-image 2D + 3D annotation JSONs ───────────────────────────
echo "[1/2] Exporting per-image annotations ..."
python3 "$ROOT/src/pipeline/nuscenes_labeled/export_annotations.py" \
    --dataroot    "$NUSCENES_ROOT" \
    --version     "$NUSCENES_VERSION" \
    --output_dir  "$ANNOTATIONS_DIR" \
    # --num_workers "$NUM_WORKERS"

# ── Step 2: aggregate annotations into per-scene object tracks ───────────
echo
echo "[2/2] Generating object tracks by scene ..."
python3 "$ROOT/src/pipeline/nuscenes_labeled/generate_tracks.py" \
    --input_dir    "$ANNOTATIONS_DIR" \
    --output_dir   "$TRACKS_DIR" \
    --organize_by  scene \
    --nusc_dataroot "$NUSCENES_ROOT" \
    --nusc_version  "$NUSCENES_VERSION"

echo
echo "=== Done ==="
echo "  Annotations : $ANNOTATIONS_DIR"
echo "  Tracks      : $TRACKS_DIR"
