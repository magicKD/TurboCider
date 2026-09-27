#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
mkdir -p build/native
SDK="$(xcrun --sdk macosx --show-sdk-path)"
MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
    /cmd LC_BUILD_VERSION/{build=1; next}
    build && /minos /{print $2; exit}
')"
DEPLOYMENT_TARGET="${TURBOCIDER_DEPLOYMENT_TARGET:-${MIN_MACOS:-15.0}}"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -fvisibility=default -isysroot "$SDK" \
    -mmacosx-version-min="$DEPLOYMENT_TARGET" \
    -I bindings/c/include -I native/core -isystem "$MLX_ROOT/include" \
    -x objective-c++ tools/native/qwen21_session_probe.mm \
    -Lbuild/native -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
    -framework Foundation -framework Metal \
    -Wl,-rpath,@executable_path -Wl,-rpath,"$MLX_ROOT/lib" \
    -o build/native/qwen21-session-probe
