# Sourced by the three Qwen LoRA replay entry points from the repository root.
build_qwen_lora_replay() {
  local variant="$1" out="$2" macro="$3" host_marker="$4"
  shift 4

  export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
  source tools/native/dependencies.sh
  mkdir -p "$out"
  local probe_source="tools/native/native_qwen_lora_${variant}_ffn_replay.cpp"
  local support_source=tools/native/native_qwen_lora_ffn_replay.hpp
  local binary="native-qwen-lora-${variant}-ffn-replay"
  local probe_sha
  probe_sha="$(shasum -a 256 "$probe_source" "$support_source" | shasum -a 256 | awk '{print $1}')"
  local common=(-std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -x objective-c++
    -mmacosx-version-min=26.2 "-D${macro}_SOURCE_SHA=\"${probe_sha}\""
    -fmodules-cache-path="$out/module-cache" -I native/core -I bindings/c/include
    -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp")

  local mode="${1:---host-only}"
  [[ "$#" -le 1 ]] || return 2
  if [[ "$mode" == --host-only ]]; then
    xcrun clang++ "${common[@]}" -c "$probe_source" -o "$out/$binary.o"
    printf '%s\n' "$host_marker=PASS device_execution=0 model_loads=0"
    return
  fi
  [[ "$mode" == --link || "$mode" == --cpu-test ]] || {
    printf 'usage: build_native_qwen_lora_%s_ffn_replay.sh [--host-only|--link|--cpu-test]\n' "$variant" >&2
    return 2
  }
  local library="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Set the matching newly built native library directory explicitly.}"
  [[ -f "$library/libturbocider.dylib" && -f "$library/runtime-build/runtime-build-manifest.json" ]] || return 1
  [[ ! -e "$out/$binary" ]] || {
    printf '%s\n' 'Refusing to overwrite a replay binary; choose a fresh owned output directory.' >&2
    return 1
  }
  local build_id
  build_id="$(sed -n 's/^  "runtime_build_id": "\(tc-runtime-build-v1-[a-f0-9]*\)",$/\1/p' "$library/runtime-build/runtime-build-manifest.json")"
  [[ "$build_id" =~ ^tc-runtime-build-v1-[a-f0-9]{64}$ ]] || return 1
  common+=("-D${macro}_EXPECTED_BUILD_ID=\"${build_id}\"")
  xcrun clang++ "${common[@]}" "$probe_source" \
    -L"$library" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
    -framework Foundation -framework Metal -framework IOSurface \
    -Wl,-rpath,"$library" -Wl,-rpath,"$MLX_ROOT/lib" \
    -o "$out/$binary"
  if [[ "$mode" == --cpu-test ]]; then
    "$out/$binary" --cpu-contract
  else
    printf '%s\n' "Built $out/$binary (not run)."
  fi
}
