#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$OUT" == /* ]] || OUT="$PWD/$OUT"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -I native/core -isystem "$MLX_ROOT/include" tools/native/qwen_down_kernel_probe.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen-down-kernel-probe"
