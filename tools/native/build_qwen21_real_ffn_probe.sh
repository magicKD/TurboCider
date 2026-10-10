#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/real-ffn-probe}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
  tools/native/qwen21_real_ffn_probe.cpp -L"$LIB" -lturbocider \
  -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-real-ffn-probe"
