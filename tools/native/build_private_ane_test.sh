#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/private-ane-tests}"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm \
  native/backends/private/ane_mil.cpp tests/native/private_ane_program_test.cpp \
  -framework Foundation -framework Metal -framework IOSurface -o "$OUT/private-ane-program-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm native/backends/private/ane_mil.cpp \
  native/backends/private/ane_executor.mm native/backends/ane_memory.cpp native/core/gguf_decode.cpp \
  tests/native/private_ane_executor_test.cpp -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/private-ane-executor-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm native/backends/private/ane_mil.cpp \
  native/backends/private/ane_executor.mm native/backends/ane_memory.cpp native/core/gguf_decode.cpp \
  tests/native/private_ane_transfer_test.mm -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/private-ane-transfer-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm native/core/gguf_decode.cpp \
  tests/native/private_ane_w8_stage_test.mm -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/private-ane-w8-stage-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror tests/native/ane_w8a8_math_test.cpp native/core/gguf_decode.cpp \
  -o "$OUT/ane-w8a8-math-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm native/backends/private/ane_mil.cpp native/core/gguf_decode.cpp \
  tests/native/private_ane_w8_pipeline_test.mm -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/private-ane-w8-pipeline-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm native/backends/private/ane_mil.cpp native/core/gguf_decode.cpp \
  tests/native/private_ane_w8_ffn_test.mm -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/private-ane-w8-ffn-test"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc \
  -mmacosx-version-min=15.0 native/backends/private/ane_program.mm native/backends/private/ane_mil.cpp \
  native/backends/private/ane_w8_executor.mm native/backends/ane_memory.cpp native/core/gguf_decode.cpp \
  tests/native/private_ane_w8_executor_test.mm -framework Foundation -framework Metal -framework IOSurface \
  -o "$OUT/private-ane-w8-executor-test"
