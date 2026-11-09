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
# Keep the unqualified FFN injection seam out of production source. The
# diagnostic executable gets a private, reproducible overlay of the six model
# files; the regular library and the packaged app are never modified.
OVERLAY="$(mktemp -d "$OUT/ltx098-hybrid-source.XXXXXX")"
cp native/models/ltx098/{block.hpp,block.cpp,transformer.hpp,transformer.cpp,pipeline.hpp,pipeline.cpp} "$OVERLAY/"
patch -s -d "$OVERLAY" -p1 < tools/native/ltx098_ane_hybrid_overlay.patch
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -DTURBOCIDER_LTX098_EXPERIMENTAL_HYBRID=1 \
 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
 -I "$OVERLAY" -I native/models/ltx098 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/ltx098_ane_hybrid_generate.mm native/components/text/umt5.cpp \
 "$OVERLAY/block.cpp" "$OVERLAY/transformer.cpp" "$OVERLAY/pipeline.cpp" \
 -L"$NATIVE" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
 -framework Foundation -framework CoreML -framework CoreVideo -framework IOSurface -framework Metal -framework AVFoundation \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/ltx098-ane-hybrid-generate"
