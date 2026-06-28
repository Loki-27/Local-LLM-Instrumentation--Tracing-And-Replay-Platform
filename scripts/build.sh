#!/usr/bin/env bash
# scripts/build.sh
#
# One-shot configure + build.
# Run from anywhere — it always resolves the project root correctly.
#
# Usage:
#   bash scripts/build.sh              # Release build (default)
#   bash scripts/build.sh --debug      # Debug build (slower, has symbols)
#   bash scripts/build.sh --clean      # Wipe build/ and rebuild from scratch

set -e   # exit on any error

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$ROOT/build"
BUILD_TYPE="Release"

# ── Parse flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --debug) BUILD_TYPE="Debug" ;;
        --clean) rm -rf "$BUILD_DIR"; echo "[build] Wiped $BUILD_DIR" ;;
    esac
done

echo "──────────────────────────────────────────────────────────"
echo " llm-tracer build"
echo " Root:       $ROOT"
echo " Build dir:  $BUILD_DIR"
echo " Build type: $BUILD_TYPE"
echo "──────────────────────────────────────────────────────────"

# ── Check submodule is initialized ────────────────────────────────────────────
if [ ! -f "$ROOT/extern/llama.cpp/CMakeLists.txt" ]; then
    echo ""
    echo "[error] extern/llama.cpp is empty."
    echo "        Initialize it with:"
    echo "          git submodule update --init --recursive"
    echo ""
    exit 1
fi

# ── CMake configure ───────────────────────────────────────────────────────────
echo ""
echo "[build] Configuring..."
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Copy compile_commands.json to root so clangd and editors pick it up
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    cp "$BUILD_DIR/compile_commands.json" "$ROOT/compile_commands.json"
fi

# ── Parallel build ────────────────────────────────────────────────────────────
# Detect CPU count — works on both macOS (sysctl) and Linux (nproc)
if command -v nproc &>/dev/null; then
    CORES=$(nproc)
elif command -v sysctl &>/dev/null; then
    CORES=$(sysctl -n hw.logicalcpu)
else
    CORES=4
fi

echo "[build] Compiling with $CORES cores..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j"$CORES"

# ── Success ───────────────────────────────────────────────────────────────────
echo ""
echo "✓ Build complete  ($BUILD_TYPE)"
echo ""
echo "  Binaries:"
echo "    ./build/llm_tracer      --model models/<file>.gguf [--prompt <text>]"
echo "    ./build/test_hook_dump  --model models/<file>.gguf"
echo ""
