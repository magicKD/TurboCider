#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-lora-model-cpu-20261008}"
mkdir -p "$OUT"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2)
xcrun clang++ "${COMMON[@]}" -fsyntax-only \
  native/models/qwen21/native_block.cpp native/models/qwen21/native_denoiser.cpp
xcrun clang++ "${COMMON[@]}" -DTURBOCIDER_ENABLE_PRIVATE_ANE -fsyntax-only \
  native/models/qwen21/native_block.cpp native/models/qwen21/native_denoiser.cpp
xcrun clang++ "${COMMON[@]}" tests/native/native_qwen_lora_denoiser_contract_test.cpp \
  -o "$OUT/native-qwen-lora-denoiser-contract"
"$OUT/native-qwen-lora-denoiser-contract"
printf '%s\n' 'native_qwen_lora_model_host=PASS configurations=public,private devices=0 models=0'
