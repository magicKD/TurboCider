#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:?fresh output required}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?matching native required}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test ! -e "$OUT/libqwen_convrot_factorized_probe.dylib"
mkdir -p "$OUT"
CPP_SHA="$(shasum -a 256 tools/profiling/qwen_base/convrot_factorized_probe.mm | awk '{print $1}')"
HEADER_SHA="$(shasum -a 256 tools/profiling/qwen_base/convrot_factorized.hpp | awk '{print $1}')"
BOUND_ID="$(.venv/bin/python -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["runtime_build_id"])' "$LIB/runtime-build/runtime-build-manifest.json")"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-missing-field-initializers -fobjc-arc -mmacosx-version-min=26.2 -dynamiclib \
  -DTC_CONVROT_FACTOR_CPP_SHA="\"$CPP_SHA\"" -DTC_CONVROT_FACTOR_HEADER_SHA="\"$HEADER_SHA\"" -DTC_CONVROT_FACTOR_NATIVE_ID="\"$BOUND_ID\"" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" tools/profiling/qwen_base/convrot_factorized_probe.mm \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/libqwen_convrot_factorized_probe.dylib"
