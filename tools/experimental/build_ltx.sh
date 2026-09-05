#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MLX_ROOT:?Set local MLX_ROOT}"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
SDK="${SDKROOT:-$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk}"
TOOLCHAIN="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin"
ROOT="$PWD/experimental/video/ltx/vendor"
OUT="$PWD/build/native/ltx"
mkdir -p "$OUT"
FLAGS=(-O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" -mmacosx-version-min=15.0 -I "$ROOT")
OBJECTS=()
for name in ltx ltx_conditioning ltx_connector ltx_transformer_io ltx_latent_stats ltx_rng ltx_blocks; do
 "$TOOLCHAIN/clang" -std=c11 "${FLAGS[@]}" -DLTX_ENABLE_ANE_MLP -DLTX_ENABLE_ANE_V2A -DLTX_ENABLE_ANE_KV -DLTX_ENABLE_ANE_QKV -c "$ROOT/$name.c" -o "$OUT/$name.o"
 OBJECTS+=("$OUT/$name.o")
done
for name in ltx_safetensors ltx_weights ltx_gpu ltx_upsampler ltx_video_vae ltx_ane_mlp ltx_ane_v2a ltx_ane_kv ltx_ane_qkv; do
 "$TOOLCHAIN/clang" -std=c11 -fobjc-arc "${FLAGS[@]}" -c "$ROOT/$name.m" -o "$OUT/$name.o"
 OBJECTS+=("$OUT/$name.o")
done
for name in ltx_mlx_upsampler ltx_mlx_video_vae; do
 "$TOOLCHAIN/clang++" -std=c++20 "${FLAGS[@]}" -isystem "$MLX_ROOT/include" -c "$ROOT/$name.cpp" -o "$OUT/$name.o"
 OBJECTS+=("$OUT/$name.o")
done
rm -f "$OUT/libltx.a"
"$TOOLCHAIN/ar" rcs "$OUT/libltx.a" "${OBJECTS[@]}"
printf 'Built native LTX component archive\n'
