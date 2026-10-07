#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/ane-public-w8-tests}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations \
 -fobjc-arc -mmacosx-version-min=26.2 native/backends/ane_gpu.mm native/backends/ane_runtime.mm \
 native/backends/ane_public_w8.mm native/backends/ane_memory.cpp native/core/gguf_decode.cpp \
 tests/native/ane_public_w8_test.mm -framework Foundation -framework CoreML -framework CoreVideo \
 -framework IOSurface -framework Metal -o "$OUT/ane-public-w8-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations \
 -fobjc-arc -mmacosx-version-min=26.2 native/backends/ane_gpu.mm native/backends/ane_runtime.mm \
 native/backends/ane_public_w8.mm native/backends/ane_memory.cpp native/core/gguf_decode.cpp \
 tests/native/ane_public_w8_executor_test.mm -framework Foundation -framework CoreML -framework CoreVideo \
 -framework IOSurface -framework Metal -o "$OUT/ane-public-w8-executor-test"
