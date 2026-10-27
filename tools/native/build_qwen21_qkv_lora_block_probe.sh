#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen21-qkv-lora-block}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
QKV_CPP_SHA="$(shasum -a 256 tools/native/qwen21_qkv_lora_block_probe.mm | awk '{print $1}')"
QKV_HEADER_SHA="$(shasum -a 256 tools/native/qwen21_qkv_lora_prepare.hpp | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2 \
  -DTC_QKV_PROBE_CPP_SHA="\"$QKV_CPP_SHA\"" -DTC_QKV_PROBE_HEADER_SHA="\"$QKV_HEADER_SHA\"" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/native/qwen21_qkv_lora_block_probe.mm \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-qkv-lora-block-probe"
