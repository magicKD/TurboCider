#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_GRAPH_OUT:-$PWD/build/native-graph-validation}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/qwen-real-ffn-20261007}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -mmacosx-version-min=26.2 -fmodules-cache-path="$OUT/module-cache" \
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
  tools/native/native_qwen_ffn_replay.cpp \
  native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_ffn.mm \
  native/backends/metal_graph/mlx_bridge.cpp \
  native/backends/metal_graph/mlx_ffn_island.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/native-qwen-ffn-replay"
