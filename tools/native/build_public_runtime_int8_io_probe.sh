#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/public-runtime-int8-io-probe}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations \
 -fobjc-arc -mmacosx-version-min=26.2 tools/native/public_runtime_int8_io_probe.mm \
 -framework Foundation -framework CoreML -framework CoreVideo -framework IOSurface -o "$OUT/probe"
