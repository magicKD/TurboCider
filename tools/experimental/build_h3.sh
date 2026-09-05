#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
SDK="${SDKROOT:-$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk}"
CC="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"
ROOT="$PWD/experimental/video/h3/vendor"
OUT="$PWD/build/native/h3"
mkdir -p "$OUT"
OBJECTS=()
for name in h3 h3_host h3_safetensors h3_weights h3_text_encoder h3_dit_schedule h3_dit h3_video_vae h3_taeh3 h3_video_encoder h3_audio_vae h3_terminal h3_vision_encoder h3_multimodal; do
 "$CC" -std=c11 -O3 -D_DARWIN_C_SOURCE -isysroot "$SDK" -mmacosx-version-min=15.0 -I "$ROOT" -c "$ROOT/$name.c" -o "$OUT/$name.o"
 OBJECTS+=("$OUT/$name.o")
done
for name in h3_metal h3_gpu h3_tokenizer h3_super h3_coreml h3_ane_bridge h3_ane_mlp h3_ane_linear; do
 "$CC" -std=c11 -O3 -fobjc-arc -D_DARWIN_C_SOURCE -isysroot "$SDK" -mmacosx-version-min=15.0 -I "$ROOT" -c "$ROOT/$name.m" -o "$OUT/$name.o"
 OBJECTS+=("$OUT/$name.o")
done
rm -f "$OUT/libh3.a"
"$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin/ar" rcs "$OUT/libh3.a" "${OBJECTS[@]}"
printf 'Built native H3 archive\n'
