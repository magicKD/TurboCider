#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
mkdir -p "$OUT"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
# Match the installed MLX deployment target; do not assume a developer path.
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
  -isystem "$MLX_ROOT/include" native/backends/ane_memory.cpp native/core/gguf_decode.cpp \
  native/backends/ane_runtime.mm tools/native/ane_runtime_probe.cpp \
  -L"$MLX_ROOT/lib" -lmlx -ljaccl -framework Foundation -framework CoreML \
  -framework CoreVideo -framework IOSurface -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/ane-runtime-probe"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
  native/backends/ane_memory.cpp native/core/gguf_decode.cpp native/backends/ane_runtime.mm \
  tools/native/ane_runtime_packed_probe.cpp -framework Foundation -framework CoreML \
  -framework CoreVideo -framework IOSurface -o "$OUT/ane-runtime-packed-probe"
