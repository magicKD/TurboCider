#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
mkdir -p build/native
xcrun clang++ -std=c++20 -O2 -Wall -Wextra tools/native/z_image_gate_norm_probe.cpp \
    -isystem "$MLX_ROOT/include" -L"$MLX_ROOT/lib" -lmlx \
    -Wl,-rpath,"$MLX_ROOT/lib" -o build/native/z-image-gate-norm-probe
xcrun clang++ -std=c++20 -O2 -Wall -Wextra tools/native/z_image_metal_probe.cpp \
    -isystem "$MLX_ROOT/include" -L"$MLX_ROOT/lib" -lmlx \
    -Wl,-rpath,"$MLX_ROOT/lib" -o build/native/z-image-metal-probe
xcrun clang++ -std=c++20 -O2 -Wall -Wextra tools/native/z_image_gemm_probe.cpp \
    -isystem "$MLX_ROOT/include" -L"$MLX_ROOT/lib" -lmlx \
    -Wl,-rpath,"$MLX_ROOT/lib" -o build/native/z-image-gemm-probe
xcrun clang++ -std=c++20 -O2 -Wall -Wextra tools/native/z_image_gpu_time_probe.cpp \
    -isystem "$MLX_ROOT/include" -isystem "$MLX_ROOT/include/metal_cpp" \
    -L"$MLX_ROOT/lib" -lmlx -framework Metal -framework Foundation \
    -Wl,-rpath,"$MLX_ROOT/lib" -o build/native/z-image-gpu-time-probe
