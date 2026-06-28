#!/usr/bin/env bash
# scripts/download_model.sh
#
# Downloads TinyLlama 1.1B Chat Q4_K_M — the dev/test model for this project.
# ~670 MB. Runs on any machine without a GPU.
#
# Usage:
#   bash scripts/download_model.sh
#
# To use a different model instead:
#   1. Download any GGUF from HuggingFace into models/
#   2. Pass --model models/<yourfile>.gguf to the binaries

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
MODEL_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/$MODEL_NAME"

mkdir -p "$MODELS_DIR"

# ── Already downloaded? ────────────────────────────────────────────────────────
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    echo "[download_model] Already exists: $MODEL_PATH ($SIZE)"
    echo "                 Nothing to do."
    exit 0
fi

# ── Download ───────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────"
echo " Downloading TinyLlama 1.1B Chat (Q4_K_M)"
echo " ~670 MB — this takes a few minutes"
echo " Destination: $MODEL_PATH"
echo "──────────────────────────────────────────────────────────"
echo ""

if command -v wget &>/dev/null; then
    wget --show-progress -O "$MODEL_PATH" "$MODEL_URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
else
    echo "[error] Neither wget nor curl is installed."
    echo "        Install one and rerun, or download manually from:"
    echo "        $MODEL_URL"
    exit 1
fi

# ── Verify ─────────────────────────────────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
    echo "[error] Download failed — file not found after download attempt."
    exit 1
fi

SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
echo ""
echo "✓ Download complete: $MODEL_PATH ($SIZE)"
echo ""
echo "  Test it:"
echo "    ./build/llm_tracer --model $MODEL_PATH --prompt \"The sky is\""
echo ""
