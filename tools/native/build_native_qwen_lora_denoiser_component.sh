#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_LORA_COMPONENT_OUT:-$PWD/build/native-lora-components-20261008}"
mkdir -p "$OUT"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include")
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || { printf '%s\n' 'usage: build_native_qwen_lora_denoiser_component.sh [--host-only]' >&2; exit 2; }
  TARGET="$OUT/native-qwen-lora-denoiser-component.o"
  [[ ! -e "$TARGET" && ! -L "$TARGET" ]] || { printf '%s\n' "Refusing to overwrite: $TARGET" >&2; exit 1; }
  xcrun clang++ "${COMMON[@]}" -c tools/native/native_qwen_lora_denoiser_component.cpp -o "$TARGET"
  printf '%s\n' 'native_lora_component_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_qwen_lora_denoiser_component.sh [--host-only]' >&2; exit 2; }
# This ABI adds runtime factors, lease loading and denoiser workspace fields.
# An explicit new full library is mandatory; never default to historical 16fec.
LIB="${TURBOCIDER_NATIVE_LORA_LIBRARY_DIR:?Set the newly built matching full native library directory}"
[[ "$LIB" = /* && -f "$LIB/libturbocider.dylib" ]] || { printf '%s\n' 'Matching absolute native library directory required' >&2; exit 1; }
TARGET="$OUT/native-qwen-lora-denoiser-component"
[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || { printf '%s\n' "Refusing to overwrite: $TARGET" >&2; exit 1; }
xcrun clang++ "${COMMON[@]}" tools/native/native_qwen_lora_denoiser_component.cpp \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$TARGET"
printf '%s\n' "Built $TARGET (not run)."
