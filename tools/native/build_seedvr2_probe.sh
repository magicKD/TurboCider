#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-$PWD/build/seedvr2}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
mkdir -p "$OUT" "$OUT/module-cache"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
export CLANG_MODULE_CACHE_PATH="$OUT/module-cache"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{build=1; next} build && /minos /{print $2; exit}')"
VAE_UPSCALE_FLAGS=(-DTURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT=0)
case "${TURBOCIDER_VAE_UPSCALE_EXPERIMENT:-0}" in
 0) ;;
 1) VAE_UPSCALE_FLAGS=(-DTURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT=1) ;;
 *) printf 'TURBOCIDER_VAE_UPSCALE_EXPERIMENT must be 0 or 1\n' >&2; exit 2 ;;
esac
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter \
  "${VAE_UPSCALE_FLAGS[@]}" \
  -mmacosx-version-min="${MLX_MIN_MACOS:-15.0}" \
  -isysroot "$SDK" -I native/core -isystem "$MLX_ROOT/include" \
  -isystem "$MLX_ROOT/include/metal_cpp" \
  native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp native/models/seedvr2/pipeline.cpp \
  tools/native/seedvr2_probe.cpp -L "$MLX_ROOT/lib" -lmlx \
  -framework Metal -framework Foundation -framework Accelerate \
  -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/seedvr2-probe"
printf '%s\n' "$OUT/seedvr2-probe"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter \
  -mmacosx-version-min="${MLX_MIN_MACOS:-15.0}" \
  -isysroot "$SDK" -I native/core -isystem "$MLX_ROOT/include" \
  -isystem "$MLX_ROOT/include/metal_cpp" native/core/common.cpp \
  tools/native/seedvr2_conv_probe.cpp -L "$MLX_ROOT/lib" -lmlx \
  -framework Metal -framework Foundation -framework Accelerate \
  -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/seedvr2-conv-probe"
printf '%s\n' "$OUT/seedvr2-conv-probe"
