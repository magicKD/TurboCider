#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="$PWD/build/native"
[[ -f "$OUT/libturbocider.dylib" ]] || { echo 'Build the native runtime first.' >&2; exit 1; }
MINIMUM="$(otool -l "$OUT/libturbocider.dylib" | awk '/minos /{print $2;exit}')"
for probe in qwen3_quant nvfp4; do
  name="${probe//_/-}"
  xcrun clang++ -std=c++20 -O2 "-mmacosx-version-min=$MINIMUM" \
    -I native/core -isystem "$MLX_ROOT/include" "tools/native/${probe}_probe.cpp" \
    -L "$OUT" -lturbocider -L "$MLX_ROOT/lib" -lmlx \
    "-Wl,-rpath,$OUT" "-Wl,-rpath,$MLX_ROOT/lib" -o "$OUT/$name-probe"
done
if [[ -n "${LLAMA_SOURCE:-}" && -n "${LLAMA_BUILD:-}" ]]; then
  xcrun clang++ -std=c++17 -O2 -I "$LLAMA_SOURCE/include" -I "$LLAMA_SOURCE/ggml/include" \
    tools/validation/llama_qwen_conditioning.cpp -L "$LLAMA_BUILD/bin" -lllama -lggml-base \
    "-Wl,-rpath,$LLAMA_BUILD/bin" -o "$OUT/llama-qwen-conditioning"
fi
