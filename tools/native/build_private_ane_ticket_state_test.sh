#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_NATIVE_OUT:-$PWD/build/private-ane-ticket-state-test}"
mkdir -p "$OUT"
# CPU only. Does not link Metal/private ANE or create a device.
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  tests/native/private_ane_ticket_state_test.cpp -o "$OUT/private-ane-ticket-state-test"
"$OUT/private-ane-ticket-state-test"
