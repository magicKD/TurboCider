#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-build/native-w1-register-20261008/stage-component}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
# Host build only. The executable defaults to no work and requires an explicit
# --cpu-only or owner-scheduled --tiny-device; this script never runs it.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=26.2 native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_ane_handoff.mm \
  native/backends/metal_graph/native_ane_stages.mm \
  native/backends/ane_gpu.mm native/backends/private/ane_program.mm \
  native/core/gguf_decode.cpp tools/native/native_ane_register_stage_component.mm \
  -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/native-ane-register-stage-component"
printf '%s\n' "Built $OUT/native-ane-register-stage-component (not run)."
