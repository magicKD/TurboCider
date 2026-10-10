#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-memory-bridge-test}"
mkdir -p "$OUT"
# Real existing ledger/Context, synthetic backing identities; no Metal/MLX
# linkage, device creation, model import or new memory-accounting implementation.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I native/core -I native/runtime \
  tests/native/native_memory_bridge_test.cpp \
  native/backends/metal_graph/memory_bridge.cpp \
  native/runtime/memory_execution.cpp native/runtime/memory_scheduler.cpp \
  native/runtime/memory_watchdog.cpp native/runtime/memory_trace.cpp \
  native/runtime/memory_accounting.cpp native/runtime/memory_plan.cpp \
  native/runtime/memory_schedule.cpp native/runtime/memory_manifest.cpp \
  native/runtime/memory_policy.cpp native/core/common.cpp \
  -o "$OUT/native-memory-bridge-test"
"$OUT/native-memory-bridge-test"
