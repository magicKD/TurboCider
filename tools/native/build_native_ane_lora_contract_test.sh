#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_ane_lora_contract_test.sh' >&2; exit 2; }
CONTRACT_OUT="${TURBOCIDER_NATIVE_ANE_LORA_CONTRACT_OUT:-$PWD/build/native-ane-lora-contract-20261008}"
mkdir -p "$CONTRACT_OUT"
CONTRACT_BINARY="$CONTRACT_OUT/native-ane-lora-contract-test"
[[ ! -e "$CONTRACT_BINARY" ]] || { printf '%s\n' 'refusing to overwrite existing contract binary; choose a fresh TURBOCIDER_NATIVE_ANE_LORA_CONTRACT_OUT' >&2; exit 2; }
# CPU-only: no Metal/MLX/private framework linkage, no device or payload reads.
xcrun clang++ -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  tests/native/native_ane_lora_contract_test.cpp -o "$CONTRACT_BINARY"
"$CONTRACT_BINARY"
