#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
NATIVE="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
if [[ "$NATIVE" != /* ]]; then NATIVE="$PWD/$NATIVE"; fi
OUT="${TURBOCIDER_LTX098_PROBE_OUT:-$NATIVE}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
test -f "$NATIVE/libturbocider.dylib"
mkdir -p "$OUT"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -isysroot "$SDK" \
 -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/ltx098_ane_bias_split_probe.mm -L"$NATIVE" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
 -framework Foundation -framework CoreML -framework CoreVideo -framework IOSurface -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/ltx098-ane-bias-split-probe"
