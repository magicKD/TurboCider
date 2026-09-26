#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
  if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  else
    export DEVELOPER_DIR="$(xcode-select -p)"
  fi
fi
mkdir -p build/native/ltx-attention-probe-build
xcrun clang -std=c11 -O3 -fobjc-arc -Wall -Wextra \
  -c native/models/ltx_runtime/ltx_gpu.m \
  -o build/native/ltx-attention-probe-build/ltx_gpu.o
xcrun clang -std=c11 -O3 -Wall -Wextra \
  tools/native/ltx_attention_probe.c build/native/ltx-attention-probe-build/ltx_gpu.o \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o build/native/ltx-attention-probe
xcrun clang -std=c11 -O3 -Wall -Wextra \
  tools/native/ltx_qkv_projection_probe.c build/native/ltx-attention-probe-build/ltx_gpu.o \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o build/native/ltx-qkv-projection-probe
xcrun clang -std=c11 -O3 -Wall -Wextra \
  tools/native/ltx_attention_branch_probe.c build/native/ltx-attention-probe-build/ltx_gpu.o \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o build/native/ltx-attention-branch-probe
