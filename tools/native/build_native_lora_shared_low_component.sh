#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
MODE="${1:---cpu-contract}"
[[ "$#" -le 1 ]] || exit 2
OUT="${TURBOCIDER_NATIVE_SHARED_LOW_OUT:-$PWD/build/native-lora-shared-low-20261008}"
mkdir -p "$OUT"
SOURCE=tools/native/native_lora_shared_low_component.cpp
SDK="$(xcrun --sdk macosx --show-sdk-path)"
COMMON=(-isysroot "$SDK" -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror)
if [[ "$MODE" == --cpu-contract ]]; then
  xcrun clang++ "${COMMON[@]}" -DTC_SHARED_LOW_CPU_ONLY=1 "$SOURCE" -o "$OUT/native-lora-shared-low-cpu"
  "$OUT/native-lora-shared-low-cpu" --cpu-contract
  exit 0
fi
[[ "$MODE" == --host-only || "$MODE" == --link ]] || {
  printf '%s\n' 'usage: build_native_lora_shared_low_component.sh [--cpu-contract|--host-only|--link]' >&2; exit 2;
}
source tools/native/dependencies.sh
SHA="$(shasum -a 256 "$SOURCE" | awk '{print $1}')"
COMMON+=(-fobjc-arc -mmacosx-version-min=26.2 -fmodules-cache-path="$OUT/module-cache"
  -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include"
  -DTC_SHARED_LOW_SOURCE_SHA=\""$SHA"\")
if [[ "$MODE" == --host-only ]]; then
  xcrun clang++ "${COMMON[@]}" -c "$SOURCE" -o "$OUT/native-lora-shared-low-component.o"
  printf '%s\n' 'shared_low_host_object=PASS device_execution=0 model_loads=0'
  exit 0
fi
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Select the fresh matching native library explicitly.}"
[[ -f "$LIB/libturbocider.dylib" && -f "$LIB/runtime-build/runtime-build-manifest.json" ]] || exit 2
[[ ! -e "$OUT/native-lora-shared-low-component" ]] || {
  printf '%s\n' 'Refusing to overwrite a component binary; choose a fresh owned output directory.' >&2; exit 2;
}
# The new split methods are absent from historical libraries. Verify the
# relevant source/header ABI against the selected library before linking.
BUILD_ID="$(python3 - "$LIB/runtime-build/runtime-build-manifest.json" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
sources={entry['path']:entry['sha256'] for entry in manifest['inputs']['sources']}
for name in ['native/backends/metal_graph/native_lora.hpp',
             'native/backends/metal_graph/native_lora.mm',
             'native/backends/metal_graph/kernels/native_lora.hpp',
             'native/backends/metal_graph/backend.hpp',
             'native/backends/metal_graph/memory_bridge.hpp']:
    if sources.get(name)!=hashlib.sha256(Path(name).read_bytes()).hexdigest():
        raise SystemExit('Fresh source/header mismatch: '+name)
identity=manifest['runtime_build_id']
if not re.fullmatch(r'tc-runtime-build-v1-[0-9a-f]{64}',identity):
    raise SystemExit('Malformed native build identity')
print(identity)
PY
)"
COMMON+=(-DTC_SHARED_LOW_EXPECTED_BUILD_ID=\""$BUILD_ID"\")
xcrun clang++ "${COMMON[@]}" "$SOURCE" \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal -framework IOSurface \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/native-lora-shared-low-component"
printf '%s\n' "Built $OUT/native-lora-shared-low-component (not run)."
