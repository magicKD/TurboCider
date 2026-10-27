#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:?fresh output directory required}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native library required}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
test ! -e "$OUT/projection-describe"
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/profiling/qwen_base/projection_describe.cpp | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -DTC_W8_DESCRIBE_CPP_SHA="\"$CPP_SHA\"" -I native/core -I bindings/c/include \
  tools/profiling/qwen_base/projection_describe.cpp -L "$LIB" -lturbocider \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/projection-describe"
