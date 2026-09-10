#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
 if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
 else export DEVELOPER_DIR="$(xcode-select -p)"
 fi
fi
source tools/native/dependencies.sh
OUT="$PWD/build/native"
mkdir -p "$OUT" "$OUT/module-cache"
export CLANG_MODULE_CACHE_PATH="$OUT/module-cache"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
TOOLCHAIN="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin"
[[ -d "$TOOLCHAIN" ]] || TOOLCHAIN="$DEVELOPER_DIR/usr/bin"
CXX="$TOOLCHAIN/clang++"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
DEPLOYMENT_TARGET="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}"
MACOS_FLAGS=(-mmacosx-version-min="$DEPLOYMENT_TARGET")
COMMON=(-std=c++20 -O2 -fobjc-arc -fvisibility=hidden -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I bindings/c/include -I native/core -isystem "$MLX_ROOT/include" -Wall -Wextra -Wno-unused-parameter)
OBJECTS=()
SOURCES=(
 native/core/common.cpp
 native/components/text/qwen3.cpp
 native/components/text/umt5.cpp
 native/components/weights/affine.cpp native/platform/apple/wan_checkpoint.mm
 native/components/diffusion/wan.cpp
 native/components/vae/taehv.cpp
 native/models/wan/dit.cpp
 native/models/wan/wan_pipeline.cpp
 native/models/wan/hybrid.cpp native/platform/apple/wan_hybrid.mm
 native/platform/apple/request.mm native/platform/apple/profile.mm native/platform/apple/tokenizer.mm
 native/platform/apple/unigram_tokenizer.mm
 native/platform/apple/device.mm native/platform/apple/results.mm
 native/platform/apple/wan_session.mm native/platform/apple/h3_session.mm native/platform/apple/ltx_session.mm
 native/platform/apple/llada_session.mm
 native/api/c_api.mm
 native/runtime/execution.cpp native/runtime/plan.cpp native/runtime/residency.cpp native/runtime/lora_identity.cpp
 native/backends/mlx.cpp native/backends/coreml.mm native/backends/artifact_cache.mm native/backends/coreml_resources.mm
 native/models/registry.cpp native/models/flux_module.cpp native/models/wan_module.cpp native/models/h3_module.cpp native/models/ltx_module.cpp native/models/z_image_module.cpp native/models/z_image_gguf_module.cpp native/models/llada_module.cpp
 native/models/z_image/gguf.cpp
 native/models/z_image/z_image.cpp
 native/models/llada/llada.cpp native/models/llada/llada_text.cpp
 native/models/llada/llada_transformer.cpp
 native/models/flux2/pipeline.cpp native/models/flux2/flux_text.cpp native/models/flux2/flux_transformer.cpp
 native/models/flux2/flux_vae.cpp native/models/flux2/flux_encode.cpp
 native/media/image.mm native/media/input.mm native/media/video.mm native/media/audio.mm
)
for src in "${SOURCES[@]}"; do
 obj="$OUT/$(basename "${src%.*}").o"
 flags=(-x c++)
 if [[ "$src" == *.mm ]]; then flags=(-x objective-c++ -fobjc-arc); fi
 "$CXX" "${COMMON[@]}" "${flags[@]}" -fvisibility=default -c "$src" -o "$obj"
 OBJECTS+=("$obj")
done
VIDEO_ROOT="$PWD/native/models/h3_runtime"
VIDEO_OUT="$OUT/h3-runtime"
mkdir -p "$VIDEO_OUT"
CC="$TOOLCHAIN/clang"
H3_OBJECTS=()
for src in h3 h3_host h3_safetensors h3_weights h3_quant_cache h3_text_encoder h3_dit_schedule h3_dit h3_streaming_policy h3_video_vae h3_taeh3 h3_video_encoder h3_audio_vae h3_terminal h3_vision_encoder h3_multimodal h3_ffmpeg h3_ane_disabled; do
 "$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$VIDEO_ROOT" -c "$VIDEO_ROOT/$src.c" -o "$VIDEO_OUT/$src.o"
 H3_OBJECTS+=("$VIDEO_OUT/$src.o")
done
for src in h3_metal h3_gpu h3_tokenizer h3_coreml; do
 "$CC" -std=c11 -O3 -fobjc-arc -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$VIDEO_ROOT" -c "$VIDEO_ROOT/$src.m" -o "$VIDEO_OUT/$src.o"
 H3_OBJECTS+=("$VIDEO_OUT/$src.o")
done
install -m 0644 "$VIDEO_ROOT/h3_shaders.metal" "$OUT/h3_shaders.metal"
LTX_ROOT="$PWD/native/models/ltx_runtime"
LTX_OUT="$OUT/ltx-runtime"
mkdir -p "$LTX_OUT"
LTX_OBJECTS=()
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I native/runtime -c native/runtime/block_residency.c -o "$LTX_OUT/block_residency.o"
LTX_OBJECTS+=("$LTX_OUT/block_residency.o")
for src in ltx ltx_conditioning ltx_connector ltx_transformer_io ltx_latent_stats ltx_rng ltx_blocks; do
 "$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -DLTX_ENABLE_ANE_MLP -DLTX_ENABLE_ANE_V2A -DLTX_ENABLE_ANE_KV -DLTX_ENABLE_ANE_QKV -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c "$LTX_ROOT/$src.c" -o "$LTX_OUT/$src.o"
 LTX_OBJECTS+=("$LTX_OUT/$src.o")
done
for src in ltx_safetensors ltx_weights ltx_gpu ltx_gemma_tokenizer ltx_gemma_encoder ltx_upsampler ltx_video_vae ltx_ane_mlp ltx_ane_v2a ltx_ane_kv ltx_ane_qkv; do
 "$CC" -std=c11 -O3 -fobjc-arc -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c "$LTX_ROOT/$src.m" -o "$LTX_OUT/$src.o"
 LTX_OBJECTS+=("$LTX_OUT/$src.o")
done
for src in ltx_mlx_upsampler ltx_mlx_video_vae ltx_mlx_audio_vae ltx_mlx_vocoder ltx_mlx_bwe ltx_video_convert; do
 "$CXX" -std=c++20 -O3 -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -isystem "$MLX_ROOT/include" -c "$LTX_ROOT/$src.cpp" -o "$LTX_OUT/$src.o"
 LTX_OBJECTS+=("$LTX_OUT/$src.o")
done
"$TOOLCHAIN/ar" rcs "$LTX_OUT/libltx-runtime.a" "${LTX_OBJECTS[@]}"
install -m 0644 "$LTX_ROOT/ltx_shaders.metal" "$OUT/ltx_shaders.metal"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_gemma_encode.c -o "$LTX_OUT/ltx_gemma_encode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_gemma_encode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-gemma-encode" -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_audio_vae_decode.c -o "$LTX_OUT/ltx_audio_vae_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_audio_vae_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-audio-vae-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,"$MLX_ROOT/lib"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_vocoder_decode.c -o "$LTX_OUT/ltx_vocoder_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_vocoder_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-vocoder-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,"$MLX_ROOT/lib"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_bwe_decode.c -o "$LTX_OUT/ltx_bwe_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_bwe_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-bwe-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,"$MLX_ROOT/lib"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_video_vae_decode.c -o "$LTX_OUT/ltx_video_vae_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_video_vae_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-video-vae-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -Wl,-rpath,"$MLX_ROOT/lib"
"$CXX" -std=c++20 -O3 -fobjc-arc -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -I native/media -c tools/native/ltx_video_finalizer.mm -o "$LTX_OUT/ltx_video_finalizer_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_video_finalizer_tool.o" "$LTX_OUT/libltx-runtime.a" "$OUT/video.o" -o "$OUT/ltx-video-finalizer" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework AVFoundation -framework CoreMedia -framework CoreVideo -Wl,-rpath,"$MLX_ROOT/lib"
"$CXX" -std=c++20 -O3 -fobjc-arc -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I native/media -c tools/native/ltx_audio_mux.mm -o "$LTX_OUT/ltx_audio_mux_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_audio_mux_tool.o" "$OUT/audio.o" "$OUT/video.o" -o "$OUT/ltx-audio-mux" -framework Foundation -framework AVFoundation -framework AudioToolbox -framework CoreMedia -framework CoreVideo
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" -dynamiclib "${OBJECTS[@]}" "${H3_OBJECTS[@]}" "$LTX_OUT/libltx-runtime.a" -o "$OUT/libturbocider.dylib" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework CoreML -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework IOSurface -framework Accelerate -framework ImageIO -framework CoreGraphics -framework UniformTypeIdentifiers -framework Vision -Wl,-rpath,"$MLX_ROOT/lib" -Wl,-install_name,@rpath/libturbocider.dylib
"$CC" -std=c11 -O3 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$VIDEO_ROOT" -c tools/native/h3_dit_streaming_probe.c -o "$VIDEO_OUT/h3_dit_streaming_probe.o"
"$CC" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$VIDEO_OUT/h3_dit_streaming_probe.o" -L"$OUT" -lturbocider -o "$OUT/h3-dit-streaming-probe" -Wl,-rpath,@executable_path
"$CC" -std=c11 -O3 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$VIDEO_ROOT" -c tools/native/h3_quantize_stream_cache.c -o "$VIDEO_OUT/h3_quantize_stream_cache.o"
"$CC" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$VIDEO_OUT/h3_quantize_stream_cache.o" -L"$OUT" -lturbocider -o "$OUT/h3-quantize-stream-cache" -Wl,-rpath,@executable_path
"$CXX" "${COMMON[@]}" -fobjc-arc apps/cli/main.mm services/turbociderd/service.mm -L"$OUT" -lturbocider -framework Foundation -Wl,-rpath,@executable_path -o "$OUT/turbocider"
printf 'Built %s\n' "$OUT/turbocider"
mkdir -p "$OUT/coreml"
export TURBOCIDER_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
tools/native/build_app.sh
