#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_VAE_PROBE_OUT:-$PWD/build/prunavaed}"
mkdir -p "$OUT"
xcrun clang -std=c11 -O3 -I native/models/ltx_runtime \
  -c tools/native/ltx_video_vae_decode.c -o "$OUT/decode.o"
xcrun clang++ -std=c++20 -O3 -mmacosx-version-min="$(turbocider_mlx_min_macos)" \
  -isystem "$MLX_ROOT/include" native/models/ltx_runtime/ltx_mlx_video_vae.cpp \
  "$OUT/decode.o" -L"$MLX_ROOT/lib" -lmlx -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/ltx-video-vae-decode"
