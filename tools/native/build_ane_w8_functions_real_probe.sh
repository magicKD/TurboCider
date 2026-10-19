#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
PROBE_OUT="${TURBOCIDER_NATIVE_OUT:?fresh probe output required}"
PROBE_LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching V7-or-newer native required}"
[[ "$PROBE_LIB" == /* ]] || PROBE_LIB="$PWD/$PROBE_LIB"
test ! -e "$PROBE_OUT/libane_w8_functions_real_probe.dylib"
mkdir -p "$PROBE_OUT"
PROBE_SHA="$(shasum -a 256 tools/profiling/qwen_base/w8_functions_real_probe.mm | awk '{print $1}')"
PROBE_NATIVE_ID="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_build_id"])' "$PROBE_LIB/runtime-build/runtime-build-manifest.json")"
[[ "$PROBE_NATIVE_ID" =~ ^tc-runtime-build-v1-[0-9a-f]{64}$ ]]
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
  -DTC_REAL_FUNCTIONS_CPP_SHA="\"$PROBE_SHA\"" -DTC_REAL_FUNCTIONS_NATIVE_ID="\"$PROBE_NATIVE_ID\"" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/profiling/qwen_base/w8_functions_real_probe.mm \
  -L "$PROBE_LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal -framework IOSurface -Wl,-rpath,"$PROBE_LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$PROBE_OUT/libane_w8_functions_real_probe.dylib"
test "$PROBE_SHA" = "$(shasum -a 256 tools/profiling/qwen_base/w8_functions_real_probe.mm | awk '{print $1}')"
