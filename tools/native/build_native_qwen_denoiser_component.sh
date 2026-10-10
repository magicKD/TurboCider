#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_GRAPH_OUT:-$PWD/build/native-graph-components-20261008}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native-graph-20261008}"
mkdir -p "$OUT"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc
  -mmacosx-version-min=26.2 -fmodules-cache-path="$OUT/module-cache"
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include")
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || { printf '%s\n' 'usage: build_native_qwen_denoiser_component.sh [--host-only]' >&2; exit 2; }
  xcrun clang++ "${COMMON[@]}" -c tools/native/native_qwen_denoiser_component.cpp \
    -o "$OUT/native-qwen-denoiser-component.o"
  printf '%s\n' 'denoiser_component_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_qwen_denoiser_component.sh [--host-only]' >&2; exit 2; }
[[ -f "$LIB/libturbocider.dylib" ]] || { printf '%s\n' "Missing current full native library: $LIB/libturbocider.dylib" >&2; exit 1; }
# Use a freshly built full library containing the matching native model/bridge
# ABI. Do not compile competing native definitions against an older library.
xcrun clang++ "${COMMON[@]}" tools/native/native_qwen_denoiser_component.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/native-qwen-denoiser-component"
printf '%s\n' "Built $OUT/native-qwen-denoiser-component (not run)."
