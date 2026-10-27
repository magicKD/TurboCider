#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:?fresh probe build directory required}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native directory required}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
test ! -e "$OUT/libqwen_fixed_program_probe.dylib"
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/profiling/qwen_base/fixed_program_probe.mm | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
  -DTC_FIXED_PROGRAM_CPP_SHA="\"$CPP_SHA\"" -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
  tools/profiling/qwen_base/fixed_program_probe.mm -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal -framework IOSurface -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/libqwen_fixed_program_probe.dylib"
