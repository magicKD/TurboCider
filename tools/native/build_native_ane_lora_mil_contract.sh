#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$#" == 1 ]] || { printf '%s\n' 'usage: build_native_ane_lora_mil_contract.sh owned-legacy-fixture-directory' >&2; exit 2; }
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
OUT="${TURBOCIDER_NATIVE_LORA_MIL_OUT:-$PWD/build/native-lora-foundation-20261008/mil-contract}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  native/backends/private/ane_mil.cpp tests/native/native_ane_lora_mil_test.cpp \
  -o "$OUT/native-ane-lora-mil-contract"
"$OUT/native-ane-lora-mil-contract" "$1"
printf '%s\n' 'native_ane_lora_mil_host=PASS compiled_programs=0 device_execution=0 models=0'
