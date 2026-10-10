#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_gate_up_contract.sh' >&2; exit 2; }
OUT="${TURBOCIDER_NATIVE_GATE_UP_OUT:-$PWD/build/native-gate-up-20261008/contracts}"
mkdir -p "$OUT"
CPU_FLAGS=(-std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror)
for TEST in native_gate_up_contract_test native_gate_up_source_test; do
  xcrun clang++ "${CPU_FLAGS[@]}" "tests/native/$TEST.cpp" -o "$OUT/$TEST"
  "$OUT/$TEST"
done
# Host syntax only: never creates a Metal device, compiles MSL, or submits work.
xcrun clang++ "${CPU_FLAGS[@]}" -fobjc-arc -mmacosx-version-min=26.2 \
  -fmodules-cache-path="$OUT/module-cache" -fsyntax-only \
  native/backends/metal_graph/native_gate_up.mm
printf '%s\n' 'native_gate_up_host=PASS device_created=0 shader_compiled=0 model_loads=0'
