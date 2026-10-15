#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-build/native-qwen-block-component}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
COMPONENT_SOURCE=tools/native/native_qwen_block_component.cpp
COMPONENT_NAME=native-qwen-block-component
if [[ "${2:-}" == "--cpu-contract" ]]; then
  COMPONENT_SOURCE=tests/native/native_qwen_block_contract_test.cpp
  COMPONENT_NAME=native-qwen-block-contract
elif [[ -n "${2:-}" ]]; then
  printf '%s\n' 'Unknown component build mode' >&2
  exit 2
fi
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -mmacosx-version-min=26.2 native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_linear.mm native/backends/metal_graph/native_lora.mm native/backends/metal_graph/native_ffn.mm \
  native/backends/metal_graph/native_attention.mm native/backends/metal_graph/native_qwen_ops.mm \
  native/models/qwen21/native_block.cpp "$COMPONENT_SOURCE" \
  -framework Foundation -framework Metal -o "$OUT/$COMPONENT_NAME"
printf '%s\n' "Built $OUT/$COMPONENT_NAME (not run)."
