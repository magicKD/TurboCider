#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/private-ane-tests}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -I native/core -isystem "$MLX_ROOT/include" tests/native/ane_channel_ffn_test.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/ane-channel-ffn-test"
