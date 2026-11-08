#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/convrot-ffn-error}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/convrot-direct-private}"
mkdir -p "$OUT"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
 -mmacosx-version-min="${MLX_MIN_MACOS:-15.0}" -I native -I native/core -isystem "$MLX_ROOT/include" \
 tools/native/convrot_ffn_error_probe.cpp -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl \
 -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/convrot-ffn-error-probe"
