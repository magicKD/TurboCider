#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/native-ane-handoff-types-test}"
mkdir -p "$OUT"
# CPU only: no Metal/private ANE linkage or device creation.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  tests/native/native_ane_handoff_types_test.cpp -o "$OUT/native-ane-handoff-types-test"
"$OUT/native-ane-handoff-types-test"
