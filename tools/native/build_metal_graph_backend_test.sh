#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-build/native-metal-graph-contract}"
mkdir -p "$OUT"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -mmacosx-version-min=14.0 native/backends/metal_graph/backend.mm \
  tests/native/metal_graph_backend_test.cpp -framework Foundation -framework Metal \
  -o "$OUT/metal-graph-backend-test"
printf '%s\n' "Built $OUT/metal-graph-backend-test (not run)."
