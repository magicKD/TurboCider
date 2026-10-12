#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/convrot-rotation-tests}"
mkdir -p "$OUT"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -mmacosx-version-min="${MLX_MIN_MACOS:-15.0}" -isystem "$MLX_ROOT/include" \
  tests/native/convrot_rotation_test.cpp -L"$MLX_ROOT/lib" -lmlx -ljaccl \
  -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/convrot-rotation-test"
