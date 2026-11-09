#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
NATIVE="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
OUT="${TURBOCIDER_WAN22_PROBE_OUT:-$PWD/build/wan22-probes}"
if [[ "$NATIVE" != /* ]]; then NATIVE="$PWD/$NATIVE"; fi
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
test -f "$NATIVE/libturbocider.dylib"
mkdir -p "$OUT"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MIN_MACOS="$(turbocider_mlx_min_macos)"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/unigram_probe.mm -L "$NATIVE" -lturbocider -framework Foundation \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/unigram-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_block_probe.cpp native/models/wan22/block.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-block-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_transformer_probe.cpp native/models/wan22/block.cpp native/models/wan22/transformer.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-transformer-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_vae_encoder_probe.cpp native/models/wan22/vae.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-vae-encoder-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_vae_decoder_probe.cpp native/models/wan22/vae.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-vae-decoder-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_vae_benchmark_probe.cpp native/models/wan22/vae.cpp native/models/wan22/tae_super.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-vae-benchmark-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_pipeline_probe.cpp native/models/wan22/block.cpp native/models/wan22/transformer.cpp native/models/wan22/vae.cpp native/models/wan22/tae_super.cpp native/models/wan22/pipeline.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-pipeline-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_text_probe.cpp native/models/wan22/text.cpp native/components/text/umt5.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-text-probe"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_pixels_to_video.mm \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-pixels-to-video"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
 tools/native/wan22_resolution_probe.mm native/models/wan22/block.cpp native/models/wan22/transformer.cpp \
 native/models/wan22/vae.cpp native/models/wan22/tae_super.cpp native/models/wan22/pipeline.cpp \
 native/models/wan22/text.cpp native/components/text/umt5.cpp \
 -L "$NATIVE" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$NATIVE" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/wan22-resolution-probe"
