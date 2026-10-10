#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-build/native-ane-handoff-component}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
# The reused legacy ane_gpu.mm still sets fastMathEnabled=NO. Its SDK
# deprecation is unrelated to the native handoff; other warnings remain errors.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=26.2 native/backends/metal_graph/backend.mm \
  native/backends/metal_graph/native_ane_handoff.mm \
  native/backends/ane_gpu.mm native/backends/private/ane_program.mm \
  tools/native/native_ane_handoff_component.cpp \
  -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/native-ane-handoff-component"
printf '%s\n' "Built $OUT/native-ane-handoff-component (not run)."
