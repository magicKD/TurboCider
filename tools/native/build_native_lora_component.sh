#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_LORA_OUT:-$PWD/build/native-lora-20261008/component}"
mkdir -p "$OUT"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc
  -mmacosx-version-min=26.2 -fmodules-cache-path="$OUT/module-cache"
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include")
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || exit 2
  xcrun clang++ "${COMMON[@]}" -c tools/native/native_lora_component.cpp \
    -o "$OUT/native-lora-component.o"
  printf '%s\n' 'probe_host_object=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_lora_component.sh [--host-only]' >&2; exit 2; }
# Weights snapshot/header ABI changed in this round. Require a deliberately
# selected same-source fresh library; never silently use historical 16fec.
[[ -n "${TURBOCIDER_NATIVE_LIBRARY_DIR:-}" ]] || {
  printf '%s\n' 'Set TURBOCIDER_NATIVE_LIBRARY_DIR to a fresh same-source native build; historical libraries are incompatible.' >&2
  exit 2
}
LIB="$TURBOCIDER_NATIVE_LIBRARY_DIR"
[[ -f "$LIB/libturbocider.dylib" ]] || { printf '%s\n' 'Fresh native library is missing.' >&2; exit 2; }
xcrun clang++ "${COMMON[@]}" tools/native/native_lora_component.cpp \
  native/backends/metal_graph/backend.mm native/backends/metal_graph/native_lora.mm \
  native/backends/metal_graph/mlx_bridge.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/native-lora-component"
printf '%s\n' 'probe_link=PASS device_execution=0 model_loads=0'
