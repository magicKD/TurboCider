#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_LORA_REPLAY_OUT:-$PWD/build/native-lora-sgmma-replay-host-20261008}"
mkdir -p "$OUT"
PROBE_SOURCE=tools/native/native_qwen_lora_sgmma_ffn_replay.cpp
PROBE_SHA="$(shasum -a 256 "$PROBE_SOURCE" | awk '{print $1}')"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -x objective-c++
  -mmacosx-version-min=26.2 -DTC_LORA_REPLAY_SOURCE_SHA=\""$PROBE_SHA"\"
  -fmodules-cache-path="$OUT/module-cache" -I native/core -I bindings/c/include
  -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp")
MODE="${1:---host-only}"
[[ "$#" -le 1 ]] || exit 2
if [[ "$MODE" == --host-only ]]; then
  xcrun clang++ "${COMMON[@]}" -c "$PROBE_SOURCE" -o "$OUT/native-qwen-lora-sgmma-ffn-replay.o"
  printf '%s\n' 'native_lora_ffn_replay_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$MODE" == --link || "$MODE" == --cpu-test ]] || {
  printf '%s\n' 'usage: build_native_qwen_lora_sgmma_ffn_replay.sh [--host-only|--link|--cpu-test]' >&2; exit 2;
}
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Set the matching newly built native library directory explicitly.}"
[[ -f "$LIB/libturbocider.dylib" && -f "$LIB/runtime-build/runtime-build-manifest.json" ]] || exit 1
[[ ! -e "$OUT/native-qwen-lora-sgmma-ffn-replay" ]] || {
  printf '%s\n' 'Refusing to overwrite a replay binary; choose a fresh owned output directory.' >&2; exit 1;
}
LIB_BUILD_ID="$(sed -n 's/^  "runtime_build_id": "\(tc-runtime-build-v1-[a-f0-9]*\)",$/\1/p' "$LIB/runtime-build/runtime-build-manifest.json")"
[[ "$LIB_BUILD_ID" =~ ^tc-runtime-build-v1-[a-f0-9]{64}$ ]] || exit 1
COMMON+=(-DTC_LORA_REPLAY_EXPECTED_BUILD_ID=\""$LIB_BUILD_ID"\")
xcrun clang++ "${COMMON[@]}" "$PROBE_SOURCE" \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal -framework IOSurface \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/native-qwen-lora-sgmma-ffn-replay"
if [[ "$MODE" == --cpu-test ]]; then
  "$OUT/native-qwen-lora-sgmma-ffn-replay" --cpu-contract
else
  printf '%s\n' "Built $OUT/native-qwen-lora-sgmma-ffn-replay (not run)."
fi
