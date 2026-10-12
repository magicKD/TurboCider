#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
OUT="${TURBOCIDER_NATIVE_HYBRID_DENOISER_OUT:-$PWD/build/native-w1-register-components-20261008}"
# This tool consumes the appended register/timing ABI. Do not inherit a stale
# generic library override from a historical component build.
LIB="$PWD/build/native-w1-register-20261008"
mkdir -p "$OUT"
SDK="$(xcrun --sdk macosx --show-sdk-path)"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off
  -mmacosx-version-min=26.2 -isysroot "$SDK" -I native/core -I bindings/c/include)
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || { printf '%s\n' 'usage: build_native_qwen_hybrid_denoiser_component.sh [--host-only]' >&2; exit 2; }
  TARGET="$OUT/native-qwen-hybrid-denoiser-component.o"
  [[ ! -e "$TARGET" && ! -L "$TARGET" ]] || { printf '%s\n' "Refusing to overwrite existing output: $TARGET" >&2; exit 1; }
  xcrun clang++ "${COMMON[@]}" -c tools/native/native_qwen_hybrid_denoiser_component.cpp \
    -o "$TARGET"
  printf '%s\n' 'native_qwen_hybrid_denoiser_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_qwen_hybrid_denoiser_component.sh [--host-only]' >&2; exit 2; }
TARGET="$OUT/native-qwen-hybrid-denoiser-component"
[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || { printf '%s\n' "Refusing to overwrite existing output: $TARGET" >&2; exit 1; }
[[ -f "$LIB/libturbocider.dylib" ]] || { printf '%s\n' "Missing matching full native library: $LIB/libturbocider.dylib" >&2; exit 1; }
# Link the freshly rebuilt complete runtime ABI. The fixture does not create
# MLX arrays or run MLX operators; the library's existing dependency is kept.
# This new binary never replaces either historical denoiser fixture.
xcrun clang++ "${COMMON[@]}" tools/native/native_qwen_hybrid_denoiser_component.cpp \
  -L"$LIB" -lturbocider -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -o "$TARGET"
printf '%s\n' "Built $TARGET (not run)."
