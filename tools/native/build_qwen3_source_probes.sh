#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/quantized-execution}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/quantized-execution}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=15.0 \
  -I native/core -isystem "$MLX_ROOT/include" tools/native/qwen3_prefill_plan_probe.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen3-prefill-plan-probe"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=15.0 \
  -I native/core -isystem "$MLX_ROOT/include" tools/native/qwen3_gguf_probe.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen3-gguf-probe"
