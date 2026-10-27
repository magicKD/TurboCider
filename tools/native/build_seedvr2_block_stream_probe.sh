#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
BLOCK_PROBE_OUT="${TURBOCIDER_BLOCK_PROBE_OUT:-$PWD/build/seedvr2-block-stream-probe-v1}"
mkdir -p "$BLOCK_PROBE_OUT"
BLOCK_TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{b=1;next} b && /minos /{print $2;exit}')"
BLOCK_FLAGS=(-std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -DTURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_EXPERIMENT=1
  -mmacosx-version-min="${BLOCK_TARGET:-26.0}" -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp"
  -L "$MLX_ROOT/lib" -lmlx -framework Foundation -framework Metal -framework Accelerate -Wl,-rpath,"$MLX_ROOT/lib")
BLOCK_SOURCES=(native/core/common.cpp native/models/seedvr2/geometry.cpp native/models/seedvr2/model.cpp
  native/runtime/memory_manifest.cpp native/runtime/streaming/canonical_encoding.cpp native/runtime/streaming/source_lease.cpp)
xcrun clang++ "${BLOCK_FLAGS[@]}" "${BLOCK_SOURCES[@]}" tools/native/seedvr2_block_stream_probe.cpp -o "$BLOCK_PROBE_OUT/seedvr2-block-stream-probe"
xcrun clang++ "${BLOCK_FLAGS[@]}" -DTURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_TEST_HOOKS=1 "${BLOCK_SOURCES[@]}" \
  tests/native/seedvr2_block_stream_test.cpp -o "$BLOCK_PROBE_OUT/seedvr2-block-stream-test"
printf '%s\n' "$BLOCK_PROBE_OUT"
