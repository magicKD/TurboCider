#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
out="${1:-build/profiling/image-startup/metal_trace.dylib}"
mkdir -p "$(dirname "$out")"
xcrun clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fno-objc-arc -dynamiclib \
    tools/profiling/image_startup/metal_trace.mm -framework Foundation -framework Metal -o "$out"
printf 'Built development-only Metal API observer: %s\n' "$out"
