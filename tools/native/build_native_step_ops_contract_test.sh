#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-step-ops-contract-test}"
mkdir -p "$OUT"
# Pure CPU contract; no Metal linkage, shader compilation or device creation.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  tests/native/native_step_ops_contract_test.cpp -o "$OUT/native-step-ops-contract-test"
"$OUT/native-step-ops-contract-test"
