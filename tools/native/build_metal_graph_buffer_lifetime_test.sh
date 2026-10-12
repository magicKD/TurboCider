#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-buffer-lifetime-test}"
mkdir -p "$OUT"
# Includes backend.mm to exercise its actual private ownership object. Metal
# is linked for unused backend symbols; main never creates a device or queue.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -mmacosx-version-min=26.2 tests/native/metal_graph_buffer_lifetime_test.mm \
  -framework Foundation -framework Metal -o "$OUT/native-buffer-lifetime-test"
"$OUT/native-buffer-lifetime-test"
