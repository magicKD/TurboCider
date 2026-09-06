#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
source tools/native/dependencies.sh
OUT="$PWD/build/native"
mkdir -p "$OUT" "$OUT/module-cache"
export CLANG_MODULE_CACHE_PATH="$OUT/module-cache"
SDK="${SDKROOT:-$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk}"
CXX="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++"
COMMON=(-std=c++20 -O2 -fvisibility=hidden -isysroot "$SDK" -mmacosx-version-min=26.2 -I bindings/c/include -I native/core -isystem "$MLX_ROOT/include" -Wall -Wextra -Wno-unused-parameter)
OBJECTS=()
SOURCES=(
 native/core/common.cpp
 native/platform/apple/request.mm
 native/api/c_api.mm
 native/platform/apple/profile.mm
 native/platform/apple/tokenizer.mm
 native/platform/apple/device.mm
 native/platform/apple/results.mm
 native/runtime/execution.cpp
 native/runtime/plan.cpp
 native/runtime/residency.cpp
 native/backends/mlx.cpp
 native/backends/coreml.mm
 native/backends/artifact_cache.mm
 native/backends/coreml_resources.mm
 native/models/registry.cpp
 native/models/flux_module.cpp
 native/models/h3_module.cpp
 native/models/ltx_module.cpp
 native/models/flux2/pipeline.cpp
 native/models/flux2/flux_text.cpp
 native/models/flux2/flux_transformer.cpp
 native/models/flux2/flux_vae.cpp
 native/models/flux2/flux_encode.cpp
 native/media/image.mm
 native/media/input.mm
)
for src in "${SOURCES[@]}"; do
 obj="$OUT/$(basename "${src%.*}").o"
 flags=(-x c++)
 if [[ "$src" == *.mm ]]; then flags=(-x objective-c++ -fobjc-arc); fi
 "$CXX" "${COMMON[@]}" "${flags[@]}" -fvisibility=default -c "$src" -o "$obj"
 OBJECTS+=("$obj")
done
"$CXX" -isysroot "$SDK" -mmacosx-version-min=26.2 -dynamiclib "${OBJECTS[@]}" -o "$OUT/libturbocider.dylib" -L"$MLX_ROOT/lib" -lmlx -ljaccl -framework Foundation -framework Metal -framework CoreML -framework ImageIO -framework CoreGraphics -framework UniformTypeIdentifiers -Wl,-rpath,"$MLX_ROOT/lib" -Wl,-install_name,@rpath/libturbocider.dylib
"$CXX" "${COMMON[@]}" -fobjc-arc apps/cli/main.mm services/turbociderd/service.mm -L"$OUT" -lturbocider -framework Foundation -Wl,-rpath,@executable_path -o "$OUT/turbocider"
printf 'Built %s\n' "$OUT/turbocider"
mkdir -p "$OUT/coreml"
cp tools/coreml/export_flux2.py "$OUT/coreml/"
tools/native/build_app.sh
