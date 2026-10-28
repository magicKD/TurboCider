#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-$PWD/build/seedvr2}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
mkdir -p "$OUT"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
TARGET="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '/cmd LC_BUILD_VERSION/{build=1; next} build && /minos /{print $2; exit}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -isysroot "$SDK" \
  -mmacosx-version-min="${TARGET:-15.0}" -I native/core \
  -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
  native/core/common.cpp native/models/seedvr2/geometry.cpp tests/native/seedvr2_input_lifetime_test.cpp \
  -L "$MLX_ROOT/lib" -lmlx -framework Metal -framework Foundation -framework Accelerate \
  -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/seedvr2-input-lifetime-tests"
