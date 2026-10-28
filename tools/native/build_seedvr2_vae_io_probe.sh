#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
# Independent output directory: never overwrites an App/library candidate.
VAE_IO_OUT="${TURBOCIDER_VAE_IO_PROBE_OUT:-$PWD/build/seedvr2-vae-ane-io-probe-v2}"
mkdir -p "$VAE_IO_OUT"
VAE_IO_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra \
  -mmacosx-version-min="${VAE_IO_TARGET:-26.0}" \
  -framework Foundation -framework CoreML -framework CoreVideo \
  tools/native/seedvr2_vae_coreml_io_probe.mm -o "$VAE_IO_OUT/seedvr2-vae-coreml-io-probe"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra \
  -mmacosx-version-min="${VAE_IO_TARGET:-26.0}" -isystem "$MLX_ROOT/include" \
  -isystem "$MLX_ROOT/include/metal_cpp" -L "$MLX_ROOT/lib" -lmlx \
  -framework Metal -framework Foundation -framework Accelerate \
  -Wl,-rpath,"$MLX_ROOT/lib" \
  tools/native/seedvr2_vae_gpu_tile_probe.mm -o "$VAE_IO_OUT/seedvr2-vae-gpu-tile-probe"
printf '%s\n' "$VAE_IO_OUT"
