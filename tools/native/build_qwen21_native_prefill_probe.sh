#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Select fresh matching native library}"
[[ -f "$LIB/libturbocider.dylib" && -f "$LIB/runtime-build/runtime-build-manifest.json" ]] || exit 1
BUILD_ID="$(sed -n 's/^  "runtime_build_id": "\(tc-runtime-build-v1-[a-f0-9]*\)",$/\1/p' "$LIB/runtime-build/runtime-build-manifest.json")"
[[ "$BUILD_ID" =~ ^tc-runtime-build-v1-[a-f0-9]{64}$ ]] || exit 1
OUT="${TURBOCIDER_NATIVE_PREFILL_PROBE_OUT:-$PWD/build/qwen21-native-prefill-probe-20261008}"
mkdir -p "$OUT"
[[ ! -e "$OUT/qwen21-native-prefill-probe" ]] || exit 2
SDK="$(xcrun --sdk macosx --show-sdk-path)"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -isysroot "$SDK" \
 -mmacosx-version-min=26.2 -fmodules-cache-path="$OUT/module-cache" \
 -DTC_EDIT_ENCODER_EXPECTED_BUILD_ID=\""$BUILD_ID"\" \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
 tools/native/qwen21_native_prefill_probe.mm -L"$LIB" -lturbocider \
 -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore -framework Foundation -framework Metal \
 -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-native-prefill-probe"
printf '%s\n' "Built $OUT/qwen21-native-prefill-probe (not run)"
