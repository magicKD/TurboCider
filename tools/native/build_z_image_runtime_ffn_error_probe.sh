#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
OUT="${TURBOCIDER_NATIVE_OUT:-$LIB}"
mkdir -p "$OUT"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
 -mmacosx-version-min="${MLX_MIN_MACOS:-15.0}" -I native -I native/core \
 -I bindings/c/include -isystem "$MLX_ROOT/include" \
 tools/native/z_image_runtime_ffn_error_probe.cpp -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl \
 -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/z-image-runtime-ffn-error-probe"
