#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
VAE_SHARED_OUT="${TURBOCIDER_VAE_SHARED_PROBE_OUT:-$PWD/build/seedvr2-vae-shared-tile-probe-v1}"
mkdir -p "$VAE_SHARED_OUT"
VAE_SHARED_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra \
  -mmacosx-version-min="${VAE_SHARED_TARGET:-26.0}" -isystem "$MLX_ROOT/include" \
  -isystem "$MLX_ROOT/include/metal_cpp" -L "$MLX_ROOT/lib" -lmlx \
  -framework Foundation -framework Metal -framework CoreML -framework CoreVideo \
  -framework IOSurface -framework Accelerate -Wl,-rpath,"$MLX_ROOT/lib" \
  tools/native/seedvr2_vae_shared_tile_probe.mm -o "$VAE_SHARED_OUT/seedvr2-vae-shared-tile-probe"
printf '%s\n' "$VAE_SHARED_OUT"
