#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen21-qkv-numeric}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/native/qwen21_qkv_numeric_probe.mm | awk '{print $1}')"
HEADER_SHA="$(shasum -a 256 tools/native/qwen21_qkv_numeric_prepare.hpp | awk '{print $1}')"
HELPER_SHA="$(shasum -a 256 tools/native/qwen21_qkv_lora_block_probe.mm | awk '{print $1}')"
LAB_SHA="$(shasum -a 256 tools/native/qwen21_qkv_lora_prepare.hpp | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2 \
  -DTC_NUMERIC_CPP_SHA="\"$CPP_SHA\"" -DTC_NUMERIC_HEADER_SHA="\"$HEADER_SHA\"" \
  -DTC_NUMERIC_HELPER_SHA="\"$HELPER_SHA\"" -DTC_NUMERIC_LAB_SHA="\"$LAB_SHA\"" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/native/qwen21_qkv_numeric_probe.mm \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-qkv-numeric-probe"
