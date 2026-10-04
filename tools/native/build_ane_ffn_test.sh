#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}"
mkdir -p "$OUT"
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:-$PWD/build/native}"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
for name in ane_ffn_test qwen21_runtime_split_test z_image_padding_test; do
    xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
      -I native/core -isystem "$MLX_ROOT/include" "tests/native/$name.cpp" \
      -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl \
      -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/${name//_/-}"
done
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
  -isysroot "$SDK" -mmacosx-version-min="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}" \
  -I native/core -isystem "$MLX_ROOT/include" tests/native/ane_runtime_receipt_test.mm \
  -L"$LIB" -lturbocider -framework Foundation -Wl,-rpath,"$LIB" -o "$OUT/ane-runtime-receipt-test"
