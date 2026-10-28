#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:?fresh output required}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native library required}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
test ! -e "$OUT/libqwen_handoff_observer.dylib"
test ! -e "$OUT/libqwen_ffn_handoff_c6144.dylib"
test ! -e "$OUT/libqwen_ffn_handoff_c5120.dylib"
test ! -e "$OUT/libqwen_ffn_handoff_packed_c6144.dylib"
test ! -e "$OUT/libqwen_ffn_handoff_packed_c5120.dylib"
mkdir -p "$OUT"
BASE_SHA="$(shasum -a 256 tools/profiling/qwen_base/observer.mm | awk '{print $1}')"
HANDOFF_SHA="$(shasum -a 256 tools/profiling/qwen_base/handoff_observer.mm | awk '{print $1}')"
CPP_SHA="$(shasum -a 256 tools/profiling/qwen_base/ffn_projection_probe.mm | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
  -DTC_PROFILE_SOURCE_SHA="\"$BASE_SHA\"" -DTC_HANDOFF_SOURCE_SHA="\"$HANDOFF_SHA\"" \
  tools/profiling/qwen_base/handoff_observer.mm -framework Foundation -framework Metal -framework QuartzCore \
  -o "$OUT/libqwen_handoff_observer.dylib"
for STRATEGY in separate packed; do
 for CHANNELS in 6144 5120; do
  EXTRA=(-UTC_FFN_HANDOFF_PACKED_GPU);SUFFIX=""
  if [[ "$STRATEGY" == packed ]]; then EXTRA=(-DTC_FFN_HANDOFF_PACKED_GPU=1);SUFFIX="packed_";fi
  xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
    -DTC_FFN_PROJECTION_CPP_SHA="\"$CPP_SHA\"" -DTC_FFN_REPLAY_CHANNELS="$CHANNELS" -DTC_FFN_HANDOFF_SCREEN=1 "${EXTRA[@]}" \
    -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
    tools/profiling/qwen_base/ffn_projection_probe.mm -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
    -framework Foundation -framework Metal -framework IOSurface -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
    -o "$OUT/libqwen_ffn_handoff_${SUFFIX}c$CHANNELS.dylib"
 done
done
