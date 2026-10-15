#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"
source tools/native/dependencies.sh
LIB="${TURBOCIDER_NATIVE_LIBRARY_DIR:?Choose matching runtime}"
OUT="${TURBOCIDER_ENCODER_RESULT_TEST_OUT:-$PWD/build/qwen21-encoder-result-test-20261008}"
mkdir -p "$OUT"
[[ ! -e "$OUT/qwen21-encoder-result-test" ]] || exit 2
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc \
 -mmacosx-version-min=26.2 -fmodules-cache-path="$OUT/module-cache" \
 -I native/core -isystem "$MLX_ROOT/include" tests/native/qwen21_encoder_result_test.mm \
 -L"$LIB" -lturbocider -L"$MLX_ROOT/lib" -lmlx -ljaccl -licucore \
 -framework Foundation -framework Metal -Wl,-rpath,"$LIB" -Wl,-rpath,"$MLX_ROOT/lib" \
 -o "$OUT/qwen21-encoder-result-test"
"$OUT/qwen21-encoder-result-test"
