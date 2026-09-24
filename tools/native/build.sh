#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
 if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
 else export DEVELOPER_DIR="$(xcode-select -p)"
 fi
fi
source tools/native/dependencies.sh
EXPERIMENTAL_PROBES="${TURBOCIDER_BUILD_EXPERIMENTAL_PROBES:-0}"
case "$EXPERIMENTAL_PROBES" in
 0|1) ;;
 *) printf 'TURBOCIDER_BUILD_EXPERIMENTAL_PROBES must be 0 or 1\n' >&2; exit 2 ;;
esac
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
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
 native/platform/apple/wan_session.mm native/platform/apple/h3_session.mm native/platform/apple/h3_mlx_session.mm native/platform/apple/ltx_session.mm
 native/platform/apple/llada_session.mm
 native/api/c_api.mm
 native/runtime/execution.cpp native/runtime/plan.cpp native/runtime/residency.cpp native/runtime/lora_identity.cpp
 native/backends/mlx.cpp native/backends/coreml.mm native/backends/artifact_cache.mm native/backends/coreml_resources.mm
 native/models/registry.cpp native/models/flux_module.cpp native/models/wan_module.cpp native/models/h3_module.cpp native/models/h3_mlx_module.cpp native/models/ltx_module.cpp native/models/z_image_module.cpp native/models/z_image_gguf_module.cpp native/models/llada_module.cpp
native/models/h3_mlx/geometry.cpp native/models/h3_mlx/vdn.cpp native/models/h3_mlx/vdn_mlx.cpp native/models/h3_mlx/vsa.cpp native/models/h3_mlx/vsa_attention.cpp native/models/h3_mlx/conditioner_math.cpp native/models/h3_mlx/conditioner.cpp native/models/h3_mlx/dit.cpp native/models/h3_mlx/pipeline.cpp native/models/h3_mlx/vae_weights.cpp native/models/h3_mlx/audio_vae.cpp native/models/h3_mlx/video_vae.cpp native/platform/apple/h3_mlx_checkpoint.mm native/platform/apple/h3_mlx_shards.mm native/platform/apple/h3_mlx_prompt_cache.mm native/platform/apple/h3_mlx_vae_config.mm
 native/models/ltx_mlx/block.cpp native/models/ltx_mlx/model.cpp native/models/ltx_mlx/native.cpp
 native/models/z_image/gguf.cpp
 native/models/z_image/z_image.cpp
 native/models/qwen21/transformer.cpp
 native/models/qwen21/hybrid.cpp
 native/models/qwen21/sequence.cpp
 native/models/qwen21/text_encoder.cpp
 native/models/qwen21/vae.cpp
 native/models/qwen21/vision.cpp
 native/models/qwen21/conditioning.cpp
 native/models/qwen21/pe_processor.cpp
 native/models/qwen21/pe_conditioning.cpp
 native/platform/apple/qwen21_prompt_rewrite.mm
 native/models/qwen21/pe_delta.cpp
 native/models/qwen21/pe_language.cpp
 native/models/qwen21/pe_sampling.cpp
 native/models/qwen21/pe_generation.cpp
 native/models/qwen21/pipeline.cpp
 native/models/qwen21_module.cpp
 native/platform/apple/z_image_weight_stream.mm
 native/models/llada/llada.cpp native/models/llada/llada_text.cpp
 native/models/llada/llada_transformer.cpp
 native/models/flux2/pipeline.cpp native/models/flux2/flux_text.cpp native/models/flux2/flux_transformer.cpp
 native/models/flux2/flux_vae.cpp native/models/flux2/flux_encode.cpp
 native/media/image.mm native/media/input.mm native/media/video.mm native/media/audio.mm
 native/media/pe_image.mm
)
for src in "${SOURCES[@]}"; do
 # Keep the relative path in the object name.  Multiple model directories
 # intentionally contain common names such as dit.cpp and pipeline.cpp.
 relative="${src%.*}"
 obj="$OUT/${relative//\//_}.o"
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
for src in ltx_safetensors ltx_weights ltx_gpu ltx_gemma_tokenizer ltx_gemma_encoder ltx_gemma_ane_mlp ltx_upsampler ltx_video_vae ltx_ane_mlp ltx_ane_v2a ltx_ane_kv ltx_ane_qkv; do
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
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_gemma_encode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-gemma-encode" -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework CoreML
if [[ "$EXPERIMENTAL_PROBES" == "1" ]]; then
 "$CC" -std=c11 -O3 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_gemma_mlp_probe.c -o "$LTX_OUT/ltx_gemma_mlp_probe_tool.o"
 "$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_gemma_mlp_probe_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-gemma-mlp-probe" -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework CoreML
 "$CC" -std=c11 -O3 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_gemma_ane_mlp_probe.c -o "$LTX_OUT/ltx_gemma_ane_mlp_probe_tool.o"
 "$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_gemma_ane_mlp_probe_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-gemma-ane-mlp-probe" -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework CoreML
fi
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_audio_vae_decode.c -o "$LTX_OUT/ltx_audio_vae_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_audio_vae_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-audio-vae-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,"$MLX_ROOT/lib"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_vocoder_decode.c -o "$LTX_OUT/ltx_vocoder_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_vocoder_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-vocoder-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,"$MLX_ROOT/lib"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_bwe_decode.c -o "$LTX_OUT/ltx_bwe_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_bwe_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-bwe-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,"$MLX_ROOT/lib"
"$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -c tools/native/ltx_video_vae_decode.c -o "$LTX_OUT/ltx_video_vae_decode_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_video_vae_decode_tool.o" "$LTX_OUT/libltx-runtime.a" -o "$OUT/ltx-video-vae-decode" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -Wl,-rpath,"$MLX_ROOT/lib"
"$CXX" -std=c++20 -O3 -fobjc-arc -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$LTX_ROOT" -I native/media -c tools/native/ltx_video_finalizer.mm -o "$LTX_OUT/ltx_video_finalizer_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_video_finalizer_tool.o" "$LTX_OUT/libltx-runtime.a" "$OUT/native_media_audio.o" "$OUT/native_media_video.o" -o "$OUT/ltx-video-finalizer" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework AVFoundation -framework AudioToolbox -framework CoreMedia -framework CoreVideo -Wl,-rpath,"$MLX_ROOT/lib"
"$CXX" -std=c++20 -O3 -fobjc-arc -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I native/media -c tools/native/ltx_audio_mux.mm -o "$LTX_OUT/ltx_audio_mux_tool.o"
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$LTX_OUT/ltx_audio_mux_tool.o" "$OUT/native_media_audio.o" "$OUT/native_media_video.o" -o "$OUT/ltx-audio-mux" -framework Foundation -framework AVFoundation -framework AudioToolbox -framework CoreMedia -framework CoreVideo
"$CXX" -isysroot "$SDK" "${MACOS_FLAGS[@]}" -dynamiclib "${OBJECTS[@]}" "${H3_OBJECTS[@]}" "$LTX_OUT/libltx-runtime.a" -o "$OUT/libturbocider.dylib" -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework CoreML -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework IOSurface -framework Accelerate -framework ImageIO -framework CoreGraphics -framework UniformTypeIdentifiers -framework Vision -Wl,-rpath,"$MLX_ROOT/lib" -Wl,-install_name,@rpath/libturbocider.dylib
"$CC" -std=c11 -O3 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$VIDEO_ROOT" -c tools/native/h3_dit_streaming_probe.c -o "$VIDEO_OUT/h3_dit_streaming_probe.o"
"$CC" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$VIDEO_OUT/h3_dit_streaming_probe.o" -L"$OUT" -lturbocider -o "$OUT/h3-dit-streaming-probe" -Wl,-rpath,@executable_path
"$CC" -std=c11 -O3 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -isysroot "$SDK" "${MACOS_FLAGS[@]}" -I "$VIDEO_ROOT" -c tools/native/h3_quantize_stream_cache.c -o "$VIDEO_OUT/h3_quantize_stream_cache.o"
"$CC" -isysroot "$SDK" "${MACOS_FLAGS[@]}" "$VIDEO_OUT/h3_quantize_stream_cache.o" -L"$OUT" -lturbocider -o "$OUT/h3-quantize-stream-cache" -Wl,-rpath,@executable_path
"$CXX" "${COMMON[@]}" -fobjc-arc apps/cli/main.mm services/turbociderd/service.mm -L"$OUT" -lturbocider -framework Foundation -Wl,-rpath,@executable_path -o "$OUT/turbocider"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_tensor_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-tensor-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_tokenizer_probe.cpp -L"$OUT" -lturbocider -framework Foundation -Wl,-rpath,@executable_path -o "$OUT/h3-mlx-tokenizer-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_conditioner_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-conditioner-probe"
if [[ "$EXPERIMENTAL_PROBES" == "1" ]]; then
 "$CXX" "${COMMON[@]}" tools/native/h3_mlx_encoder_benchmark_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-encoder-benchmark-probe"
 "$CXX" "${COMMON[@]}" tools/native/h3_mlx_mlp_hybrid_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-mlp-hybrid-probe"
fi
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_prompt_cache_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-prompt-cache-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_pipeline_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-pipeline-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_vsa_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-vsa-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_vsa_pipeline_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-vsa-pipeline-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_vsa_e2e_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-vsa-e2e-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_linear_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-linear-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_vdn_solve_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-vdn-solve-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_vdn_attention_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-vdn-attention-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_vdn_e2e_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-vdn-e2e-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_audio_vae_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-audio-vae-probe"
"$CXX" "${COMMON[@]}" tools/native/h3_mlx_video_vae_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/h3-mlx-video-vae-probe"
"$CXX" "${COMMON[@]}" tools/native/ltx_mlx_block_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/ltx-mlx-block-probe"
"$CXX" "${COMMON[@]}" tools/native/ltx_mlx_model_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/ltx-mlx-model-probe"
printf 'Built %s\n' "$OUT/turbocider"
"$CXX" "${COMMON[@]}" tools/native/qwen21_transformer_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-transformer-probe"
# This probe catches tc::Cancelled across the dylib boundary; its RTTI must
# have the same visibility as the native library's exception type.
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen21_text_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-text-probe"
"$CXX" "${COMMON[@]}" tools/native/qwen21_vae_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-vae-probe"
if [[ "$EXPERIMENTAL_PROBES" == "1" ]]; then
 "$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen21_session_probe.mm -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-session-probe"
 "$CXX" "${COMMON[@]}" tools/native/qwen21_mlp_hybrid_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-mlp-hybrid-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_multimodal_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-multimodal-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_multimodal_sample_probe.mm -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-multimodal-sample-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_edit_features_probe.mm -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-edit-features-probe"
fi
"$CXX" "${COMMON[@]}" tools/native/qwen21_generate.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-generate"
"$CXX" "${COMMON[@]}" tools/native/qwen21_vision_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-vision-probe"
"$CXX" "${COMMON[@]}" tools/native/qwen21_conditioning_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-conditioning-probe"
"$CXX" "${COMMON[@]}" tests/native/qwen21_media_schedule_test.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-media-schedule-test"
"$CXX" "${COMMON[@]}" tests/native/qwen21_hybrid_merge_test.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-hybrid-merge-test"
"$CXX" "${COMMON[@]}" tests/native/qwen21_prompt_rewrite_test.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-prompt-rewrite-test"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_delta_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-delta-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_language_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-language-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_vision_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-vision-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_image_decode_probe.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-image-decode-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_tokenizer_probe.mm -L"$OUT" -lturbocider -framework Foundation -Wl,-rpath,@executable_path -o "$OUT/qwen35-tokenizer-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tools/native/qwen35_pe_sample_probe.mm -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-pe-sample-probe"
"$CXX" "${COMMON[@]}" -fvisibility=default tests/native/qwen35_sampling_test.cpp -L"$OUT" -lturbocider -Wl,-rpath,@executable_path -o "$OUT/qwen35-sampling-test"
"$CXX" "${COMMON[@]}" -fvisibility=default tests/native/qwen35_generation_test.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -framework Foundation -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-generation-test"
"$CXX" "${COMMON[@]}" -fvisibility=default tests/native/qwen35_conditioning_test.cpp -L"$OUT" -lturbocider -L"$MLX_ROOT/lib" -lmlx -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen35-conditioning-test"
if [[ "${TURBOCIDER_NATIVE_ONLY:-0}" == "1" ]]; then
 exit 0
fi
mkdir -p "$OUT/coreml"
export TURBOCIDER_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
tools/native/build_app.sh
