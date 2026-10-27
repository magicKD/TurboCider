#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:?fresh output required}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native library required}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
CHANNELS="${1:?explicit channels required}";shift
[[ "$CHANNELS" == 5120 || "$CHANNELS" == 6144 ]]
[[ "$#" -gt 0 ]]
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/profiling/qwen_base/ffn_projection_probe.mm | awk '{print $1}')"
PRECISION="${TURBOCIDER_FFN_GATE_PRECISION:-bf16}"
EXTRA=(-UTC_FFN_GATE_FP16);SUFFIX=""
if [[ "$PRECISION" == fp16 ]]; then EXTRA=(-DTC_FFN_GATE_FP16=1);SUFFIX="fp16_";else [[ "$PRECISION" == bf16 ]];fi
for POLICY in "$@"; do
  case "$POLICY" in
    m16n64d) BM=16;BN=64;STATIC=0;; m16n128d) BM=16;BN=128;STATIC=0;;
    m32n64d) BM=32;BN=64;STATIC=0;; m32n128d) BM=32;BN=128;STATIC=0;;
    m64n64d) BM=64;BN=64;STATIC=0;; m64n128d) BM=64;BN=128;STATIC=0;;
    m16n64s) BM=16;BN=64;STATIC=1;; m16n128s) BM=16;BN=128;STATIC=1;;
    m32n64s) BM=32;BN=64;STATIC=1;; m32n128s) BM=32;BN=128;STATIC=1;;
    m64n64s) BM=64;BN=64;STATIC=1;; m64n128s) BM=64;BN=128;STATIC=1;;
    *) exit 2;;
  esac
  test ! -e "$OUT/libqwen_ffn_down_${SUFFIX}${POLICY}_c$CHANNELS.dylib"
  xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
    -DTC_FFN_PROJECTION_CPP_SHA="\"$CPP_SHA\"" -DTC_FFN_REPLAY_CHANNELS="$CHANNELS" -DTC_FFN_HANDOFF_SCREEN=1 -DTC_FFN_DOWN_SCREEN=1 \
    -DTC_FFN_DOWN_POLICY="\"$POLICY\"" -DTC_FFN_DOWN_BM="$BM" -DTC_FFN_DOWN_BN="$BN" -DTC_FFN_DOWN_STATIC="$STATIC" "${EXTRA[@]}" \
    -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
    tools/profiling/qwen_base/ffn_projection_probe.mm -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
    -framework Foundation -framework Metal -framework IOSurface -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
    -o "$OUT/libqwen_ffn_down_${SUFFIX}${POLICY}_c$CHANNELS.dylib"
done
