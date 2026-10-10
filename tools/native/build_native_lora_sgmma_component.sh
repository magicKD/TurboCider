#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_SGMMA_OUT:-$PWD/build/native-lora-sgmma-component-20261008}"
mkdir -p "$OUT"
SOURCE=tools/native/native_lora_sgmma_component.cpp
SHA="$(shasum -a 256 "$SOURCE" | awk '{print $1}')"
CONTRACT_SHA="$(shasum -a 256 tools/native/native_lora_sgmma_probe_contract.hpp | awk '{print $1}')"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -mmacosx-version-min=26.2
  -fmodules-cache-path="$OUT/module-cache" -I native/core -I bindings/c/include
  -isystem "$MLX_ROOT/include" -DTC_LORA_PROBE_SOURCE_SHA=\""$SHA"\"
  -DTC_LORA_PROBE_CONTRACT_SHA=\""$CONTRACT_SHA"\")
if [[ "${1:-}" == "--cpu-contract" ]]; then
  [[ "$#" == 1 ]] || exit 2
  xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
    tests/native/native_lora_sgmma_probe_contract_test.cpp -o "$OUT/native-lora-sgmma-cpu-contract"
  "$OUT/native-lora-sgmma-cpu-contract"
  exit 0
fi
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || exit 2
  xcrun clang++ "${COMMON[@]}" -c "$SOURCE" -o "$OUT/native-lora-sgmma-component.o"
  printf '%s\n' 'lora_sgmma_probe_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_lora_sgmma_component.sh [--host-only|--cpu-contract]' >&2; exit 2; }
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Select the freshly rebuilt matching runtime explicitly; historical libraries have incompatible LoRA ABI.}"
[[ -f "$LIB/libturbocider.dylib" && -f "$LIB/runtime-build/runtime-build-manifest.json" ]] || exit 2
[[ ! -e "$OUT/native-lora-sgmma-component" ]] || { printf '%s\n' 'Refusing to overwrite a probe binary. Select a new owned output directory.' >&2; exit 2; }
BUILD_ID="$(sed -n 's/^  "runtime_build_id": "\(tc-runtime-build-v1-[a-f0-9]*\)",$/\1/p' "$LIB/runtime-build/runtime-build-manifest.json")"
[[ "$BUILD_ID" =~ ^tc-runtime-build-v1-[a-f0-9]{64}$ ]] || exit 2
COMMON+=(-DTC_LORA_PROBE_EXPECTED_BUILD_ID=\""$BUILD_ID"\")
xcrun clang++ "${COMMON[@]}" "$SOURCE" \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/native-lora-sgmma-component"
printf '%s\n' "Built $OUT/native-lora-sgmma-component (not run)."
