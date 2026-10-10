#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
OUT="${TURBOCIDER_NATIVE_HYBRID_OUT:-$PWD/build/native-scale-cache-controller-20261008}"
mkdir -p "$OUT"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations
  -ffp-contract=off -fobjc-arc -mmacosx-version-min=26.2
  -I native/core -I bindings/c/include -fmodules-cache-path="$OUT/module-cache")
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || { printf '%s\n' 'usage: build_native_hybrid_ffn_component.sh [--host-only]' >&2; exit 2; }
  xcrun clang++ "${COMMON[@]}" -fsyntax-only tools/native/native_hybrid_ffn_component.cpp
  printf '%s\n' 'native_hybrid_ffn_component_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_hybrid_ffn_component.sh [--host-only]' >&2; exit 2; }
[[ ! -e "$OUT/native-hybrid-ffn-component" ]] || {
  printf '%s\n' 'Refusing to replace a component binary; choose a fresh owned output directory.' >&2; exit 1;
}
# Every source belongs to the independent native path. No MLX, model checkpoint,
# Python inference runtime or preexisting libturbocider ABI is linked.
xcrun clang++ "${COMMON[@]}" \
  tools/native/native_hybrid_ffn_component.cpp \
  native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_ffn.mm \
  native/backends/metal_graph/native_step_ops.mm \
  native/backends/metal_graph/native_ane_stages.mm \
  native/backends/metal_graph/native_ane_handoff.mm \
  native/backends/metal_graph/native_hybrid_ffn.mm \
  native/backends/ane_gpu.mm \
  native/backends/private/ane_program.mm \
  native/backends/private/ane_mil.cpp \
  native/runtime/memory_accounting.cpp \
  -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/native-hybrid-ffn-component"
printf '%s\n' "Built $OUT/native-hybrid-ffn-component (not run)."
