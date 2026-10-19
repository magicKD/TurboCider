#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
PROBE_OUT="${TURBOCIDER_NATIVE_OUT:?fresh probe build directory required}"
PROBE_LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching V7-or-newer native required}"
[[ "$PROBE_LIB" == /* ]] || PROBE_LIB="$PWD/$PROBE_LIB"
test -f "$PROBE_LIB/libturbocider.dylib"
test ! -e "$PROBE_OUT/libane_w8_functions_probe.dylib"
mkdir -p "$PROBE_OUT"
PROBE_SHA="$(shasum -a 256 tools/profiling/qwen_base/w8_functions_probe.mm | awk '{print $1}')"
PROBE_NATIVE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_build_id"])' "$PROBE_LIB/runtime-build/runtime-build-manifest.json")"
[[ "$PROBE_NATIVE_ID" =~ ^tc-runtime-build-v1-[0-9a-f]{64}$ ]]
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
  -DTC_W8_FUNCTION_CPP_SHA="\"$PROBE_SHA\"" -DTC_W8_FUNCTION_NATIVE_ID="\"$PROBE_NATIVE_ID\"" -I bindings/c/include \
  tools/profiling/qwen_base/w8_functions_probe.mm -L "$PROBE_LIB" -lturbocider \
  -framework Foundation -framework Metal -Wl,-rpath,"$PROBE_LIB" \
  -o "$PROBE_OUT/libane_w8_functions_probe.dylib"
test "$PROBE_SHA" = "$(shasum -a 256 tools/profiling/qwen_base/w8_functions_probe.mm | awk '{print $1}')"
