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
OVERLAY="$(mktemp -d "$OUT/ltx098-vae-source.XXXXXX")"
cp native/models/ltx098/vae.cpp "$OVERLAY/vae.cpp"
patch -s -d "$OVERLAY" -p1 < tools/native/ltx098_vae_pad_baseline.patch
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
COMMON=(-std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}"
 -I native/models/ltx098 -I tools/native -I native/core -I bindings/c/include
 -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp")
LINK=(-L"$NATIVE" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -framework Metal
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib")
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_decode_benchmark.cpp "$OVERLAY/vae.cpp" "${LINK[@]}" -o "$OUT/ltx098-vae-decode-baseline"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_decode_benchmark.cpp native/models/ltx098/vae.cpp "${LINK[@]}" -o "$OUT/ltx098-vae-decode-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_decoder_probe.cpp "$OVERLAY/vae.cpp" "${LINK[@]}" -o "$OUT/ltx098-vae-small-baseline"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_decoder_probe.cpp native/models/ltx098/vae.cpp "${LINK[@]}" -o "$OUT/ltx098-vae-small-candidate"
# Link both end-to-end probes from the same source set; only the decoder pad
# differs. This avoids depending on dynamic-library symbol interposition.
PIPELINE=(native/components/text/umt5.cpp native/models/ltx098/block.cpp
 native/models/ltx098/transformer.cpp native/models/ltx098/pipeline.cpp)
xcrun clang++ "${COMMON[@]}" -fobjc-arc tools/native/ltx098_generate_probe.mm "${PIPELINE[@]}" \
 "$OVERLAY/vae.cpp" "${LINK[@]}" -framework AVFoundation -o "$OUT/ltx098-generate-pad-baseline"
xcrun clang++ "${COMMON[@]}" -fobjc-arc tools/native/ltx098_generate_probe.mm "${PIPELINE[@]}" \
 native/models/ltx098/vae.cpp "${LINK[@]}" -framework AVFoundation -o "$OUT/ltx098-generate-pad-candidate"
