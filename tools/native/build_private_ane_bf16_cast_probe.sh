#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/ane-bf16-cast}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off \
 -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=15.0 \
 native/backends/ane_gpu.mm native/backends/private/ane_program.mm native/core/gguf_decode.cpp \
 tools/native/private_ane_bf16_cast_probe.cpp -framework Foundation -framework Metal -framework IOSurface \
 -o "$OUT/probe"
