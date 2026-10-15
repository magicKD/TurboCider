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
  xcrun clang++ "${COMMON[@]}" -c tools/native/native_qwen_block_replay.cpp \
    -o "$OUT/native-qwen-block-replay.o"
  printf '%s\n' 'probe_host_object=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_qwen_block_replay.sh [--host-only]' >&2; exit 2; }
# Header/source and this library must be from the same observer-enabled build;
# rebuilding only the probe against an old Session ABI is never a runnable setup.
xcrun clang++ "${COMMON[@]}" \
  tools/native/native_qwen_block_replay.cpp \
  native/models/qwen21/native_block.cpp \
  native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_ffn.mm \
  native/backends/metal_graph/native_linear.mm native/backends/metal_graph/native_lora.mm \
  native/backends/metal_graph/native_qwen_ops.mm \
  native/backends/metal_graph/native_attention.mm \
  native/backends/metal_graph/mlx_bridge.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/native-qwen-block-replay"
