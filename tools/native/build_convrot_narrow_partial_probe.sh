#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/convrot-narrow-partial}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
 -I native/core -isystem "$MLX_ROOT/include" tools/native/convrot_narrow_partial_probe.cpp \
 -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
 -o "$OUT/convrot-narrow-partial-probe"
