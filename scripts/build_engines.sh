#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
JOBS=${JOBS:-8}
MLX_ROOT=${MLX_ROOT:-${TURBOCIDER_MLX_ROOT:-}}

cd "$ROOT"

echo "== H3 native engine =="
make -C engines/h3 -j"$JOBS" h3

echo "== LTX native core and tools =="
make -C engines/ltx-mac -j"$JOBS" test tools

if [ -n "$MLX_ROOT" ]; then
    echo "== LTX MLX + ANE pipeline =="
    make -C engines/ltx-mac -j"$JOBS" \
        MLX_ROOT="$MLX_ROOT" \
        mlx-ane-pipeline
else
    echo "== LTX MLX + ANE pipeline skipped: set TURBOCIDER_MLX_ROOT or MLX_ROOT =="
fi

echo "== FLUX.2 native bridge =="
PYTHON=${PYTHON:-python3}
"$PYTHON" engines/flux2/engine/scripts/build_native.py \
    --output-dir engines/flux2/engine/build

echo "== TurboCider Python tests =="
PYTHONPATH="$ROOT/src" "$PYTHON" -m pytest -q tests

echo "All engine builds completed."
