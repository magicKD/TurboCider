#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
SPATIAL_PROBE_OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-$PWD/build/seedvr2-spatial-probe-v2}"
mkdir -p "$SPATIAL_PROBE_OUT"
SPATIAL_PROBE_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
SPATIAL_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -mmacosx-version-min="${SPATIAL_PROBE_TARGET:-26.0}"
  -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" -L "$MLX_ROOT/lib" -lmlx
  -framework Foundation -framework Metal -framework Accelerate -Wl,-rpath,"$MLX_ROOT/lib")
case "${TURBOCIDER_VAE_UPSCALE_EXPERIMENT:-0}" in
 0) ;;
 1) SPATIAL_FLAGS+=(-DTURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT=1) ;;
 *) printf 'TURBOCIDER_VAE_UPSCALE_EXPERIMENT must be 0 or 1\n' >&2; exit 2 ;;
esac
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp tests/native/seedvr2_spatial_files_test.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-spatial-files-test"
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_spatial_decode_probe.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-spatial-decode-probe"
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tests/native/seedvr2_spatial_cancel_test.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-spatial-cancel-test"
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_spatial_encode_probe.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-spatial-encode-probe"
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tests/native/seedvr2_spatial_encode_cancel_test.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-spatial-encode-cancel-test"
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tests/native/seedvr2_spatial_encode_stream_test.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-spatial-encode-stream-test"
xcrun clang++ "${SPATIAL_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tests/native/seedvr2_upscale_history_test.cpp -o "$SPATIAL_PROBE_OUT/seedvr2-upscale-history-test"
printf '%s\n' "$SPATIAL_PROBE_OUT"
