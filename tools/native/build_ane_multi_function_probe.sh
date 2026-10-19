#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
PROBE_OUT="${TURBOCIDER_NATIVE_OUT:?fresh probe build directory required}"
PROBE_LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native directory required}"
[[ "$PROBE_LIB" == /* ]] || PROBE_LIB="$PWD/$PROBE_LIB"
test -f "$PROBE_LIB/libturbocider.dylib"
test ! -e "$PROBE_OUT/libane_multi_function_probe.dylib"
mkdir -p "$PROBE_OUT"
PROBE_SHA="$(shasum -a 256 tools/profiling/qwen_base/multi_function_probe.mm | awk '{print $1}')"
PROBE_HEADER_SHA="$(shasum -a 256 tools/profiling/qwen_base/multi_function.hpp | awk '{print $1}')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
  -DTC_MULTI_FUNCTION_CPP_SHA="\"$PROBE_SHA\"" -DTC_MULTI_FUNCTION_HEADER_SHA="\"$PROBE_HEADER_SHA\"" \
  -I bindings/c/include -isystem "$MLX_ROOT/include" \
  tools/profiling/qwen_base/multi_function_probe.mm -L "$PROBE_LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl \
  -framework Foundation -framework Metal -framework IOSurface -Wl,-rpath,"$PROBE_LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$PROBE_OUT/libane_multi_function_probe.dylib"
test "$PROBE_SHA" = "$(shasum -a 256 tools/profiling/qwen_base/multi_function_probe.mm | awk '{print $1}')"
test "$PROBE_HEADER_SHA" = "$(shasum -a 256 tools/profiling/qwen_base/multi_function.hpp | awk '{print $1}')"
