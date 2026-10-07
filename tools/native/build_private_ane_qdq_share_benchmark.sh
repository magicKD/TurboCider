#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/private-ane-qdq-share-benchmark}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations \
 -fobjc-arc -mmacosx-version-min=15.0 native/backends/private/ane_program.mm \
 native/backends/private/ane_mil.cpp native/core/gguf_decode.cpp \
 tools/native/private_ane_qdq_share_benchmark.mm -framework Foundation \
 -framework Metal -framework IOSurface -o "$OUT/probe"
