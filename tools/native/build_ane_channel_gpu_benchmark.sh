#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/private-ane-tests}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -isystem "$MLX_ROOT/include" tools/native/ane_channel_gpu_benchmark.cpp \
  -L"$MLX_ROOT/lib" -lmlx -ljaccl -framework Foundation -framework Metal \
  -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/ane-channel-gpu-benchmark"
