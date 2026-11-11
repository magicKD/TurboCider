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
python3 tools/native/embed_qwen21_cider_kernels.py --output build/native/ltx-attention-probe-build/qwen21_cider_kernels_generated.hpp
xcrun clang++ -std=c++20 -O3 -fobjc-arc -I build/native/ltx-attention-probe-build \
  -I native/models/ltx_runtime \
  -c native/platform/apple/ltx_w8a8.mm \
  -o build/native/ltx-attention-probe-build/ltx_w8a8.o
xcrun clang -std=c11 -O3 -fobjc-arc -Wall -Wextra \
  -c native/models/ltx_runtime/ltx_gpu.m \
  -o build/native/ltx-attention-probe-build/ltx_gpu.o
xcrun clang -std=c11 -O3 -Wall -Wextra \
  tools/native/ltx_attention_probe.c build/native/ltx-attention-probe-build/ltx_gpu.o build/native/ltx-attention-probe-build/ltx_w8a8.o -lc++ \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o build/native/ltx-attention-probe
xcrun clang -std=c11 -O3 -Wall -Wextra \
  tools/native/ltx_qkv_projection_probe.c build/native/ltx-attention-probe-build/ltx_gpu.o build/native/ltx-attention-probe-build/ltx_w8a8.o -lc++ \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o build/native/ltx-qkv-projection-probe
xcrun clang -std=c11 -O3 -Wall -Wextra \
  tools/native/ltx_attention_branch_probe.c build/native/ltx-attention-probe-build/ltx_gpu.o build/native/ltx-attention-probe-build/ltx_w8a8.o -lc++ \
  -framework Foundation -framework Metal -framework MetalPerformanceShaders \
  -framework MetalPerformanceShadersGraph -o build/native/ltx-attention-branch-probe
