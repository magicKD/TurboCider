#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-$PWD/build/ltx-optimization-tests}"
mkdir -p "$OUT"
DEPLOYMENT_TARGET="${TURBOCIDER_DEPLOYMENT_TARGET:-$(turbocider_mlx_min_macos)}"
xcrun clang -std=c11 -O2 -Wall -Wextra -Werror \
 -I native/models/ltx_runtime tests/native/ltx_sol_route_stats_test.c \
 -o "$OUT/ltx-sol-route-stats-test"
for component in fusion conv; do
 xcrun clang++ -std=c++20 -O3 -mmacosx-version-min="$DEPLOYMENT_TARGET" \
  -I native/models/ltx_runtime -isystem "$MLX_ROOT/include" \
  "tests/native/ltx_prunavaed_${component}_test.cpp" \
  -L "$MLX_ROOT/lib" -lmlx -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/ltx-prunavaed-${component}-test"
done
printf 'Built Sol route statistics, exact PrunaVAED fusion and numerical Conv3D tests in %s\n' "$OUT"
