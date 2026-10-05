#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/private-ane-calibration-tests}"
mkdir -p "$OUT"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
  -isystem "$MLX_ROOT/include" native/backends/private/ane_program.mm \
  native/backends/private/ane_mil.cpp native/backends/private/ane_calibration.mm native/core/gguf_decode.cpp native/backends/ane_memory.cpp \
  tests/native/private_ane_calibration_test.mm -L"$MLX_ROOT/lib" -lmlx -ljaccl \
  -framework Foundation -framework Metal -framework IOSurface -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/private-ane-calibration-test"
