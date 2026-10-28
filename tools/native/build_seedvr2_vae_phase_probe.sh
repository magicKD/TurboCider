#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
PHASE_PROBE_OUT="${TURBOCIDER_VAE_PHASE_PROBE_OUT:-$PWD/build/seedvr2-vae-phase-probe-v1}"
mkdir -p "$PHASE_PROBE_OUT"
PHASE_PROBE_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
PHASE_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -DTURBOCIDER_ENABLE_VAE_PHASE_PROFILE=1
  -mmacosx-version-min="${PHASE_PROBE_TARGET:-26.0}" -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp"
  -L "$MLX_ROOT/lib" -lmlx -framework Foundation -framework Metal -framework Accelerate -Wl,-rpath,"$MLX_ROOT/lib")
case "${TURBOCIDER_VAE_UPSCALE_EXPERIMENT:-0}" in
 0) ;;
 1) PHASE_FLAGS+=(-DTURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT=1) ;;
 *) printf 'TURBOCIDER_VAE_UPSCALE_EXPERIMENT must be 0 or 1\n' >&2; exit 2 ;;
esac
xcrun clang++ "${PHASE_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_spatial_decode_probe.cpp \
  -o "$PHASE_PROBE_OUT/seedvr2-spatial-decode-probe"
printf '%s\n' "$PHASE_PROBE_OUT"
