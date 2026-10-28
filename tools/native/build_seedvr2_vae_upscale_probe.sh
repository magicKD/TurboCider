#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
UP_PROBE_OUT="${TURBOCIDER_UPSCALE_PROBE_OUT:-$PWD/build/seedvr2-vae-upscale-probe-v1}"
mkdir -p "$UP_PROBE_OUT"
UP_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
UP_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -I native/core -DTURBOCIDER_ENABLE_VAE_UPSCALE_EXPERIMENT=1 -mmacosx-version-min="${UP_TARGET:-26.0}"
    -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" -L "$MLX_ROOT/lib" -lmlx
    -framework Foundation -framework Metal -framework Accelerate -Wl,-rpath,"$MLX_ROOT/lib")
xcrun clang++ "${UP_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
    native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_vae_upscale_capture.cpp \
    -o "$UP_PROBE_OUT/seedvr2-vae-upscale-capture"
xcrun clang++ "${UP_FLAGS[@]}" native/core/common.cpp tests/native/seedvr2_upscale_test.cpp \
    -o "$UP_PROBE_OUT/seedvr2-upscale-test"
if [[ "${TURBOCIDER_UPSCALE_CAPTURE_ONLY:-0}" != "1" ]]; then
    xcrun clang++ "${UP_FLAGS[@]}" -fobjc-arc -framework CoreML -framework CoreVideo -framework IOSurface \
        tools/native/seedvr2_vae_upscale_probe.mm -o "$UP_PROBE_OUT/seedvr2-vae-upscale-probe"
fi
printf '%s\n' "$UP_PROBE_OUT"
