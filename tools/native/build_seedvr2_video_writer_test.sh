#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-$PWD/build/seedvr2}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
mkdir -p "$OUT"
xcrun clang++ -std=c++20 -O2 -fobjc-arc -Wall -Wextra \
 native/media/video.mm tests/native/seedvr2_video_writer_test.cpp \
 -framework Foundation -framework AVFoundation -framework CoreVideo -framework CoreMedia \
 -o "$OUT/seedvr2-video-writer-test"
printf '%s\n' "$OUT/seedvr2-video-writer-test"
