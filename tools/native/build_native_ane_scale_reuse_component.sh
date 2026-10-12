#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-build/native-w1-scale-reuse-20261008/stage-component}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
# Host build only; never execute --tiny-device from this script.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=26.2 native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_ane_handoff.mm \
  native/backends/metal_graph/native_ane_stages.mm \
  native/backends/ane_gpu.mm native/backends/private/ane_program.mm \
  native/core/gguf_decode.cpp tools/native/native_ane_scale_reuse_component.mm \
  -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/native-ane-scale-reuse-component"
printf '%s\n' "Built $OUT/native-ane-scale-reuse-component (not run)."
