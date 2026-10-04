#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
test -f build/native/libturbocider.dylib
mkdir -p build/native
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
    -mmacosx-version-min="${MLX_MIN_MACOS:-15.0}" \
    -I native -I native/core -isystem "$MLX_ROOT/include" \
    tools/native/convrot_ffn_probe.cpp -L build/native -lturbocider \
    -L "$MLX_ROOT/lib" -lmlx -Wl,-rpath,"$PWD/build/native" \
    -Wl,-rpath,"$MLX_ROOT/lib" -o build/native/convrot-ffn-probe
