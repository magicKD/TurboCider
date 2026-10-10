#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen21-gguf-source-v1}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/local512-qwen-generation-phases-v1-private}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -I native -I native/core -isystem "$MLX_ROOT/include" tools/native/qwen21_gguf_import_probe.cpp \
  native/runtime/streaming/gguf_packed_bank.cpp native/core/gguf_affine.cpp native/core/gguf_decode.cpp \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-gguf-import-probe"
