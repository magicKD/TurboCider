#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-scale-cache-contract-20261008}"
mkdir -p "$OUT"
# CPU metadata/state only: no Metal, MLX or private ANE linkage/device creation.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  tests/native/native_weight_scale_cache_contract_test.cpp \
  -o "$OUT/native-weight-scale-cache-contract-test"
"$OUT/native-weight-scale-cache-contract-test"
