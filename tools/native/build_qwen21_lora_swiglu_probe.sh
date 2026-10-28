#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen21-lora-swiglu}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
test ! -e "$OUT/qwen21-lora-swiglu-probe"
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/native/qwen21_lora_swiglu_probe.mm | awk '{print $1}')"
HEADER_SHA="$(shasum -a 256 tools/native/qwen21_lora_swiglu_prepare.hpp | awk '{print $1}')"
HELPER_SHA="$(shasum -a 256 tools/native/qwen21_qkv_lora_block_probe.mm | awk '{print $1}')"
UNARY_SHA="$(shasum -a 256 "$MLX_ROOT/include/mlx/backend/metal/kernels/unary_ops.h" | awk '{print $1}')"
BF16_SHA="$(shasum -a 256 "$MLX_ROOT/include/mlx/backend/metal/kernels/bf16_math.h" | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2 \
  -DTC_SWIGLU_CPP_SHA="\"$CPP_SHA\"" -DTC_SWIGLU_HEADER_SHA="\"$HEADER_SHA\"" -DTC_SWIGLU_HELPER_SHA="\"$HELPER_SHA\"" \
  -DTC_SWIGLU_UNARY_SHA="\"$UNARY_SHA\"" -DTC_SWIGLU_BF16_SHA="\"$BF16_SHA\"" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/native/qwen21_lora_swiglu_probe.mm \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-lora-swiglu-probe"
