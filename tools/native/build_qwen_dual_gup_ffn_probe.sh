#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:?fresh output required}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native directory required}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
CHANNELS="${TURBOCIDER_DUAL_GUP_CHANNELS:-6144}"
[[ "$CHANNELS" == 6144 || "$CHANNELS" == 5632 ]]
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/profiling/qwen_base/ffn_projection_probe.mm | awk '{print $1}')"
HEADER_SHA="$(shasum -a 256 tools/profiling/qwen_base/dual_gup.hpp | awk '{print $1}')"
BOUND_ID="$(.venv/bin/python -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_build_id"])' "$LIB/runtime-build/runtime-build-manifest.json")"
for kind in register shared; do
  SHARED=0
  [[ "$kind" != shared ]] || SHARED=1
  test ! -e "$OUT/libqwen_dual_gup_${kind}_c${CHANNELS}.dylib"
  xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
    -DTC_FFN_PROJECTION_CPP_SHA="\"$CPP_SHA\"" -DTC_FFN_DUAL_HEADER_SHA="\"$HEADER_SHA\"" -DTC_FFN_SHARE_NATIVE_ID="\"$BOUND_ID\"" \
    -DTC_FFN_SHARE_SCREEN=1 -DTC_FFN_HANDOFF_SCREEN=1 -DTC_FFN_DUAL_SCREEN=1 -DTC_FFN_DUAL_SHARED="$SHARED" \
    -DTC_FFN_REPLAY_CHANNELS="$CHANNELS" -DTC_FFN_DUAL_BM=32 -DTC_FFN_DUAL_BN=64 \
    -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/profiling/qwen_base/ffn_projection_probe.mm \
    -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework IOSurface \
    -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/libqwen_dual_gup_${kind}_c${CHANNELS}.dylib"
done
