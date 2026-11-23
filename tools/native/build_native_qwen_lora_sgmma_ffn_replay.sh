#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/build_native_qwen_lora_ffn_replay_common.sh

build_qwen_lora_replay sgmma \
  "${TURBOCIDER_NATIVE_LORA_REPLAY_OUT:-$PWD/build/native-lora-sgmma-replay-host-20261008}" \
  TC_LORA_REPLAY native_lora_ffn_replay_host "$@"
