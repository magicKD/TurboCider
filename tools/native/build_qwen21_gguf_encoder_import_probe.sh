#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen21-gguf-encoder-import}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/qwen21-gguf-rowdecode-v1-private}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
  -I native -I native/core -isystem "$MLX_ROOT/include" tools/native/qwen21_gguf_encoder_import_probe.cpp \
  -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-gguf-encoder-import-probe"
