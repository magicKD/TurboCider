#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
OUT="${TURBOCIDER_NATIVE_HYBRID_REPLAY_OUT:-$PWD/build/native-replay-compare-cpu-20261008}"
mkdir -p "$OUT"
PROBE_SOURCE=tools/native/native_qwen_hybrid_ffn_replay.cpp
PROBE_SHA="$(shasum -a 256 "$PROBE_SOURCE" | awk '{print $1}')"
COMMON=(-std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -x objective-c++
  -mmacosx-version-min=26.2 -DTURBOCIDER_ENABLE_PRIVATE_ANE=1
  -DTC_NATIVE_REPLAY_SOURCE_SHA=\""$PROBE_SHA"\"
  -fmodules-cache-path="$OUT/module-cache" -I native/core -I bindings/c/include
  -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp")
if [[ "${1:-}" == "--host-only" ]]; then
  [[ "$#" == 1 ]] || exit 2
  xcrun clang++ "${COMMON[@]}" -c "$PROBE_SOURCE" -o "$OUT/native-qwen-hybrid-ffn-replay.o"
  printf '%s\n' 'native_qwen_hybrid_ffn_replay_host=PASS device_execution=0 model_loads=0'
  exit 0
fi
[[ "$#" == 0 ]] || { printf '%s\n' 'usage: build_native_qwen_hybrid_ffn_replay.sh [--host-only]' >&2; exit 2; }
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Set the matching current private native library directory explicitly (build/native-w1-scale-cache-20261008).}"
[[ -f "$LIB/libturbocider.dylib" && -f "$LIB/runtime-build/runtime-build-manifest.json" ]] || {
  printf '%s\n' 'Build the matching current private native library first.' >&2; exit 1;
}
[[ ! -e "$OUT/native-qwen-hybrid-ffn-replay" ]] || {
  printf '%s\n' 'Refusing to replace a replay binary; choose a new owned output directory.' >&2; exit 1;
}
LIB_BUILD_ID="$(sed -n 's/^  "runtime_build_id": "\(tc-runtime-build-v1-[a-f0-9]*\)",$/\1/p' "$LIB/runtime-build/runtime-build-manifest.json")"
[[ "$LIB_BUILD_ID" =~ ^tc-runtime-build-v1-[a-f0-9]{64}$ ]] || {
  printf '%s\n' 'Missing or invalid native runtime build identity.' >&2; exit 1;
}
COMMON+=(-DTC_NATIVE_REPLAY_EXPECTED_BUILD_ID=\""$LIB_BUILD_ID"\")
# Link the freshly rebuilt complete library once; no duplicate Backend/ledger
# definitions and no stale Session/Completion ABI from historical libraries.
xcrun clang++ "${COMMON[@]}" "$PROBE_SOURCE" \
  -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
  -framework Foundation -framework Metal -framework IOSurface \
  -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
  -o "$OUT/native-qwen-hybrid-ffn-replay"
printf '%s\n' "Built $OUT/native-qwen-hybrid-ffn-replay (not run)."
