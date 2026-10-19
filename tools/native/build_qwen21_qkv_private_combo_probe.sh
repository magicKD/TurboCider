#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen21-qkv-private-combo}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
COMBO_CPP_SHA="$(shasum -a 256 tools/native/qwen21_qkv_private_combo_probe.mm | awk '{print $1}')"
COMBO_QKV_SHA="$(shasum -a 256 tools/native/qwen21_qkv_lora_prepare.hpp | awk '{print $1}')"
COMBO_PREFIX_SHA="$(shasum -a 256 tools/native/qwen21_static_lora_prefix.hpp | awk '{print $1}')"
COMBO_MIL_SHA="$(shasum -a 256 tools/native/qwen21_static_lora_mil.hpp | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2 \
  -DTC_COMBO_CPP_SHA="\"$COMBO_CPP_SHA\"" -DTC_COMBO_QKV_SHA="\"$COMBO_QKV_SHA\"" \
  -DTC_COMBO_PREFIX_SHA="\"$COMBO_PREFIX_SHA\"" -DTC_COMBO_MIL_SHA="\"$COMBO_MIL_SHA\"" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/native/qwen21_qkv_private_combo_probe.mm \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework IOSurface \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-qkv-private-combo-probe"
