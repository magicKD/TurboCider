#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-memory-plan-test}"
mkdir -p "$OUT"
# Link the actual NativeBlock workspace formula and its host dependencies.
# The fixture never constructs Backend or invokes any Metal/MLX device API.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -mmacosx-version-min=26.2 -I native/core -I native/runtime \
  tests/native/native_memory_plan_test.cpp native/models/qwen21/native_memory_plan.cpp \
  native/models/qwen21/native_block.cpp \
  native/backends/metal_graph/backend.mm native/backends/metal_graph/native_linear.mm \
  native/backends/metal_graph/native_lora.mm \
  native/backends/metal_graph/native_ffn.mm native/backends/metal_graph/native_attention.mm \
  native/backends/metal_graph/native_qwen_ops.mm native/backends/metal_graph/memory_bridge.cpp \
  native/runtime/memory_execution.cpp native/runtime/memory_scheduler.cpp \
  native/runtime/memory_watchdog.cpp native/runtime/memory_trace.cpp \
  native/runtime/memory_accounting.cpp native/runtime/memory_plan.cpp \
  native/runtime/memory_schedule.cpp native/runtime/memory_manifest.cpp \
  native/runtime/memory_policy.cpp native/core/common.cpp \
  -framework Foundation -framework Metal -o "$OUT/native-memory-plan-test"
"$OUT/native-memory-plan-test"
