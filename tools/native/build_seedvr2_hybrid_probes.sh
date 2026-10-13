#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="$PWD/build/seedvr2"
mkdir -p "$OUT/module-cache"
export CLANG_MODULE_CACHE_PATH="$OUT/module-cache"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{build=1; next} build && /minos /{print $2; exit}')"
xcrun clang++ -std=c++20 -O2 -isysroot "$SDK" -mmacosx-version-min="$TARGET" \
  -I native/core -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
  tools/native/coreml_partitions_probe.cpp -L build/native -lturbocider -L "$MLX_ROOT/lib" -lmlx \
  -framework Foundation -framework Metal -Wl,-rpath,"$PWD/build/native" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/coreml-partitions-probe"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -isysroot "$SDK" -mmacosx-version-min="$TARGET" \
  tools/native/coreml_compute_plan_probe.mm -framework Foundation -framework CoreML \
  -o "$OUT/coreml-compute-plan-probe"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Wno-unused-parameter \
  -isysroot "$SDK" -mmacosx-version-min="$TARGET" -I native/core \
  -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
  native/core/common.cpp native/models/seedvr2/geometry.cpp native/models/seedvr2/model.cpp \
  native/models/seedvr2/hybrid.cpp native/platform/apple/seedvr2_hybrid.mm \
  native/platform/apple/seedvr2_prefix_buffers.mm \
  tools/native/seedvr2_hybrid_probe.cpp -L build/native -lturbocider -L "$MLX_ROOT/lib" -lmlx \
  -framework Foundation -framework Metal -framework CoreML -framework Accelerate \
  -Wl,-rpath,"$PWD/build/native" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/seedvr2-hybrid-probe"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Wno-unused-parameter \
  -isysroot "$SDK" -mmacosx-version-min="$TARGET" -I native/core \
  -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
  native/core/common.cpp native/models/seedvr2/geometry.cpp native/models/seedvr2/model.cpp \
  native/models/seedvr2/vae.cpp native/models/seedvr2/pipeline.cpp \
  native/models/seedvr2/hybrid.cpp native/platform/apple/seedvr2_hybrid.mm \
  native/platform/apple/seedvr2_prefix_buffers.mm \
  tools/native/seedvr2_hybrid_restore_probe.cpp -L build/native -lturbocider -L "$MLX_ROOT/lib" -lmlx \
  -framework Foundation -framework Metal -framework CoreML -framework Accelerate \
  -Wl,-rpath,"$PWD/build/native" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/seedvr2-hybrid-restore-probe"
