#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
DEPLOYMENT_TARGET="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}"
mkdir -p "$OUT"
FLAGS=(-std=c++20 -O2 -mmacosx-version-min="$DEPLOYMENT_TARGET" -isysroot "$SDK"
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp")
LINK=(-L "$OUT" -lturbocider -L "$MLX_ROOT/lib" -lmlx -Wl,-rpath,"$OUT" -Wl,-rpath,"$MLX_ROOT/lib")
xcrun clang++ "${FLAGS[@]}" tools/native/tiny_vae_probe.cpp "${LINK[@]}" -o "$OUT/tiny-vae-probe"
xcrun clang++ "${FLAGS[@]}" tests/native/tiny_preview_test.cpp "${LINK[@]}" -o "$OUT/tiny-preview-test"
