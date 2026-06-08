#!/usr/bin/env bash
# Create and populate the vqa-ad-ctu-env virtual environment.
#
# Usage (run from the project root):
#   bash scripts/setup_env.sh
#
# The venv is created in the current directory as ./vqa-ad-ctu-env/
# Activate afterwards with:
#   source vqa-ad-ctu-env/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/vqa-ad-ctu-env"

# ── Python 3.11 ───────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Install Python 3.11 or set PYTHON=/path/to/python3.11" >&2
    exit 1
fi
echo "Using $($PYTHON --version) at $(command -v "$PYTHON")"

# ── Create venv ───────────────────────────────────────────────────────────
if [ -d "$VENV" ]; then
    echo "venv already exists at $VENV — skipping creation."
else
    echo "Creating venv at $VENV ..."
    "$PYTHON" -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip wheel setuptools

# ── PyTorch 2.5.1 + CUDA 12.1 ────────────────────────────────────────────
echo
echo "Installing PyTorch 2.5.1 (CUDA 12.1)..."
"$PIP" install \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Main requirements ─────────────────────────────────────────────────────
echo
echo "Installing requirements.txt ..."
"$PIP" install -r "$ROOT/requirements.txt"

# ── Project package (editable) ────────────────────────────────────────────
echo
echo "Installing project package (editable) ..."
"$PIP" install -e "$ROOT"

# ── flash-attn (optional, needs matching CUDA/torch build) ────────────────
echo
if [ "${SKIP_FLASH_ATTN:-0}" = "1" ]; then
    echo "Skipping flash-attn (SKIP_FLASH_ATTN=1)."
else
    echo "Installing flash-attn 2.6.3 (this may take a few minutes) ..."
    "$PIP" install flash-attn==2.6.3 --no-build-isolation || {
        echo "WARNING: flash-attn installation failed."
        echo "  Install manually after activating the venv:"
        echo "    pip install flash-attn==2.6.3 --no-build-isolation"
    }
fi

echo
echo "=== Done ==="
echo "Activate with:  source vqa-ad-ctu-env/bin/activate"
