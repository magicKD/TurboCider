#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
SOURCE_PROBE_OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-$PWD/build/seedvr2-stream-input-probe-v1}"
mkdir -p "$SOURCE_PROBE_OUT"
SOURCE_PROBE_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
SOURCE_PROBE_FLAGS=(-std=c++20 -O2 -fobjc-arc -Wall -Wextra -mmacosx-version-min="${SOURCE_PROBE_TARGET:-26.0}"
  -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" -L "$MLX_ROOT/lib" -lmlx
  -framework Foundation -framework Metal -framework Accelerate -Wl,-rpath,"$MLX_ROOT/lib")
xcrun clang++ "${SOURCE_PROBE_FLAGS[@]}" native/core/common.cpp native/media/rgb_video_reader.mm \
  native/media/video_input.mm native/media/video.mm native/models/seedvr2/media.cpp tests/native/seedvr2_video_reader_test.mm \
  -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework CoreGraphics -o "$SOURCE_PROBE_OUT/seedvr2-video-reader-test"
xcrun clang++ "${SOURCE_PROBE_FLAGS[@]}" native/core/common.cpp native/models/seedvr2/geometry.cpp \
  native/models/seedvr2/model.cpp native/models/seedvr2/vae.cpp tools/native/seedvr2_encode_source_probe.cpp \
  -o "$SOURCE_PROBE_OUT/seedvr2-encode-source-probe"
printf '%s\n' "$SOURCE_PROBE_OUT"
