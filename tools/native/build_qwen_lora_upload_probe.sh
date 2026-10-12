#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/qwen-lora-upload}"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
[[ "$LIB" == /* ]] || LIB="$PWD/$LIB"
test -f "$LIB/libturbocider.dylib"
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 -fobjc-arc \
 -I native/core -isystem "$MLX_ROOT/include" tools/native/qwen_lora_upload_probe.mm \
 -L "$LIB" -lturbocider -L "$MLX_ROOT/lib" -lmlx -framework Foundation -framework Metal -framework IOSurface \
 -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen-lora-upload-probe"
