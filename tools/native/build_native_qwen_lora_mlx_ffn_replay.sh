#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/build_native_qwen_lora_ffn_replay_common.sh

build_qwen_lora_replay mlx \
  "${TURBOCIDER_NATIVE_MLX_FFN_REPLAY_OUT:-$PWD/build/native-lora-mlx-ffn-replay-host-20261008}" \
  TC_MLX_FFN_REPLAY native_lora_mlx_ffn_replay_host "$@"
