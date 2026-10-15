#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# == 2 ]] || { printf 'Usage: bash %s PROFILE CHECKPOINT_IDENTITY\n' "$0" >&2; exit 2; }
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Choose matching runtime}"
OUT="${TURBOCIDER_ENCODER_PROFILE_TEST_OUT:-$PWD/build/qwen21-encoder-profile-test}"
mkdir -p "$OUT"
[[ ! -e "$OUT/qwen21-encoder-profile-test" ]] || exit 2
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.2 \
 -I native/core -I bindings/c/include -isystem "$MLX_ROOT/include" \
 tests/native/qwen21_encoder_profile_test.cpp -L"$LIB" -lturbocider \
 -L"$MLX_ROOT/lib" -lmlx -ljaccl -framework Foundation -framework Metal \
 -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" -o "$OUT/qwen21-encoder-profile-test"
"$OUT/qwen21-encoder-profile-test" "$1" "$2"
