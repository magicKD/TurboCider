#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/build/native/vision-feature-distance"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
CXX="${CXX:-$(xcrun --find clang++)}"
mkdir -p "$(dirname "$OUT")"

"$CXX" \
    -isysroot "$SDK" \
    -std=c++20 \
    -O2 \
    -fobjc-arc \
    -Wall -Wextra \
    tools/native/vision_feature_distance.mm \
    -framework Foundation \
    -framework CoreGraphics \
    -framework ImageIO \
    -framework Vision \
    -o "$OUT"

printf 'Built %s\n' "$OUT"
