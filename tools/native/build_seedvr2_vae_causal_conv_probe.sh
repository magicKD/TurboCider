#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
CAUSAL_PROBE_OUT="${TURBOCIDER_VAE_CAUSAL_PROBE_OUT:-$PWD/build/seedvr2-vae-causal-probe-v1}"
mkdir -p "$CAUSAL_PROBE_OUT"
CAUSAL_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
CAUSAL_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -fobjc-arc
  -DTURBOCIDER_ENABLE_VAE_CAUSAL_CONV_EXPERIMENT=1 -DTURBOCIDER_ENABLE_VAE_CAUSAL_CONV_PROBE=1
  -mmacosx-version-min="${CAUSAL_TARGET:-26.0}" -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp"
  -L "$MLX_ROOT/lib" -lmlx -framework Foundation -framework Metal -framework Accelerate -framework CoreML -framework CoreVideo
  -framework IOSurface -Wl,-rpath,"$MLX_ROOT/lib")
case "${TURBOCIDER_VAE_CAUSAL_PHASE_PROFILE:-0}" in
 0) ;;
 1) CAUSAL_FLAGS+=(-DTURBOCIDER_ENABLE_VAE_PHASE_PROFILE=1) ;;
 *) printf 'TURBOCIDER_VAE_CAUSAL_PHASE_PROFILE must be 0 or 1\n' >&2; exit 2 ;;
esac
xcrun clang++ "${CAUSAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_vae_ane_causal_conv.mm \
  tools/native/seedvr2_spatial_decode_probe.cpp -o "$CAUSAL_PROBE_OUT/seedvr2-spatial-decode-probe"
xcrun clang++ "${CAUSAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_vae_ane_causal_conv.mm \
  tests/native/seedvr2_vae_ane_causal_conv_test.cpp -o "$CAUSAL_PROBE_OUT/seedvr2-vae-ane-causal-conv-test"
xcrun clang++ "${CAUSAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_vae_ane_causal_conv.mm \
  tools/native/seedvr2_vae_causal_decode_bench.cpp -o "$CAUSAL_PROBE_OUT/seedvr2-vae-causal-decode-bench"
printf '%s\n' "$CAUSAL_PROBE_OUT"
