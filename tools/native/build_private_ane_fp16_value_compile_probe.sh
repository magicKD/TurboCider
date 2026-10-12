#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/ane-fp16-value-composition}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations \
 -fobjc-arc -mmacosx-version-min=15.0 native/backends/ane_gpu.mm native/backends/private/ane_program.mm \
 native/backends/private/ane_mil.cpp tools/native/private_ane_fp16_value_compile_probe.cpp \
 -framework Foundation -framework Metal -framework IOSurface -o "$OUT/compile-probe"
