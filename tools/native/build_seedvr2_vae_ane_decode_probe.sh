#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
ANE_DECODE_OUT="${TURBOCIDER_VAE_ANE_DECODE_OUT:-$PWD/build/seedvr2-vae-ane-decode-probe-v1}"
mkdir -p "$ANE_DECODE_OUT"
ANE_DECODE_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
ANE_DECODE_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -fobjc-arc
  -DTURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT=1 -DTURBOCIDER_ENABLE_VAE_ANE_PROJECTION_PROBE=1
  -mmacosx-version-min="${ANE_DECODE_TARGET:-26.0}"
  -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" -L "$MLX_ROOT/lib" -lmlx
  -framework Foundation -framework Metal -framework Accelerate -framework CoreML -framework CoreVideo -framework IOSurface
  -Wl,-rpath,"$MLX_ROOT/lib")
xcrun clang++ "${ANE_DECODE_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_vae_ane_projection.mm \
  tools/native/seedvr2_spatial_decode_probe.cpp -o "$ANE_DECODE_OUT/seedvr2-spatial-decode-probe"
xcrun clang++ "${ANE_DECODE_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_vae_ane_projection.mm \
  tests/native/seedvr2_vae_ane_projection_test.cpp -o "$ANE_DECODE_OUT/seedvr2-vae-ane-projection-test"
printf '%s\n' "$ANE_DECODE_OUT"
