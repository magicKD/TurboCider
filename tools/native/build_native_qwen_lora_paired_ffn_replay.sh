#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/build_native_qwen_lora_ffn_replay_common.sh

build_qwen_lora_replay paired \
  "${TURBOCIDER_NATIVE_PAIRED_REPLAY_OUT:-$PWD/build/native-lora-paired-replay-host-20261008}" \
  TC_PAIRED_REPLAY native_lora_paired_ffn_replay_host "$@"
