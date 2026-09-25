#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
mkdir -p build/native
xcrun clang++ -std=c++20 -O2 -Wall -Wextra tools/native/qwen21_gemm_probe.cpp \
    -isystem "$MLX_ROOT/include" -L"$MLX_ROOT/lib" -lmlx \
    -Wl,-rpath,"$MLX_ROOT/lib" -o build/native/qwen21-gemm-probe
