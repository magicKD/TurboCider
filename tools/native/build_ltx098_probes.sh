#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
test -f "$OUT/libturbocider.dylib"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
COMMON=(-std=c++20 -O2 -fobjc-arc -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp")
LINK=(-L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib")
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_generate_probe.mm "${LINK[@]}" -framework AVFoundation -o "$OUT/ltx098-generate-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_eval_benchmark.cpp "${LINK[@]}" -o "$OUT/ltx098-eval-benchmark"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_transformer_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-transformer-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_decoder_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-vae-decoder-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_encoder_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-vae-encoder-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_block_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-block-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_scheduler_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-scheduler-probe"
xcrun clang++ "${COMMON[@]}" tools/native/umt5_probe.cpp "${LINK[@]}" -o "$OUT/umt5-probe"
xcrun clang++ "${COMMON[@]}" tools/native/unigram_probe.mm "${LINK[@]}" -o "$OUT/unigram-probe"
# Compile the current media source directly, so image-preprocessing parity can
# be checked before rebuilding the whole application library.
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_image_probe.mm native/media/input.mm "${LINK[@]}" -framework ImageIO -framework CoreGraphics -o "$OUT/ltx098-image-probe"
# Independent candidate, linked with the current LTX sources and the existing
# runtime library. This is not the shipped application until a full rebuild.
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_block_probe.cpp native/models/ltx098/block.cpp "${LINK[@]}" -o "$OUT/ltx098-block-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_transformer_probe.cpp native/models/ltx098/block.cpp native/models/ltx098/transformer.cpp "${LINK[@]}" -o "$OUT/ltx098-transformer-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_eval_benchmark.cpp native/models/ltx098/block.cpp native/models/ltx098/transformer.cpp "${LINK[@]}" -o "$OUT/ltx098-eval-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_cache_benchmark.cpp native/models/ltx098/block.cpp native/models/ltx098/transformer.cpp "${LINK[@]}" -o "$OUT/ltx098-cache-benchmark"
xcrun clang++ "${COMMON[@]}" tools/native/umt5_probe.cpp native/components/text/umt5.cpp "${LINK[@]}" -o "$OUT/umt5-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_encoder_probe.cpp native/models/ltx098/vae.cpp "${LINK[@]}" -o "$OUT/ltx098-vae-encoder-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_vae_decoder_probe.cpp native/models/ltx098/vae.cpp "${LINK[@]}" -o "$OUT/ltx098-vae-decoder-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_generate_probe.mm native/components/text/umt5.cpp native/models/ltx098/block.cpp native/models/ltx098/transformer.cpp native/models/ltx098/vae.cpp native/models/ltx098/scheduler.cpp native/models/ltx098/pipeline.cpp native/models/ltx_runtime/ltx_mlx_upsampler.cpp "${LINK[@]}" -framework AVFoundation -o "$OUT/ltx098-generate-candidate"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_cache_sequence_probe.mm native/components/text/umt5.cpp native/models/ltx098/block.cpp native/models/ltx098/transformer.cpp native/models/ltx098/vae.cpp native/models/ltx098/scheduler.cpp native/models/ltx098/pipeline.cpp native/models/ltx_runtime/ltx_mlx_upsampler.cpp "${LINK[@]}" -framework AVFoundation -o "$OUT/ltx098-cache-sequence-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_activation_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-activation-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_addcmul_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-addcmul-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_video_encode_probe.mm "${LINK[@]}" -framework AVFoundation -o "$OUT/ltx098-video-encode-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_qkv_screen.cpp "${LINK[@]}" -o "$OUT/ltx098-qkv-screen"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_rms_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-rms-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_silu_probe.cpp "${LINK[@]}" -o "$OUT/ltx098-silu-probe"
xcrun clang++ "${COMMON[@]}" tools/native/ltx098_rope_screen.cpp "${LINK[@]}" -o "$OUT/ltx098-rope-screen"
