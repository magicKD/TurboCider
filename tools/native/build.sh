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
COMMON=(-std=c++20 -O2 -fobjc-arc -fvisibility=hidden -isysroot "$SDK" -mmacosx-version-min=26.2 -I bindings/c/include -I native/core -isystem "$MLX_ROOT/include" -Wall -Wextra -Wno-unused-parameter)
OBJECTS=()
SOURCES=(
 native/core/json.mm native/core/api.mm native/core/tokenizer.mm native/core/profile.mm
 native/backends/mlx.mm native/backends/coreml.mm native/backends/artifact_cache.mm native/backends/coreml_resources.mm
 native/models/registry.mm native/models/recipes.mm native/models/flux_module.mm
 native/models/h3_module.mm native/models/ltx_module.mm
 native/models/flux.mm native/models/flux_text.mm native/models/flux_transformer.mm
 native/models/flux_vae.mm native/models/flux_encode.mm
 native/media/image.mm native/media/input.mm
)
for src in "${SOURCES[@]}"; do
 obj="$OUT/$(basename "${src%.mm}").o"
 "$CXX" "${COMMON[@]}" -fvisibility=default -c "$src" -o "$obj"
 OBJECTS+=("$obj")
done
"$CXX" -isysroot "$SDK" -mmacosx-version-min=26.2 -dynamiclib "${OBJECTS[@]}" -o "$OUT/libturbocider.dylib" -L"$MLX_ROOT/lib" -lmlx -ljaccl -framework Foundation -framework Metal -framework CoreML -framework ImageIO -framework CoreGraphics -framework UniformTypeIdentifiers -Wl,-rpath,"$MLX_ROOT/lib" -Wl,-install_name,@rpath/libturbocider.dylib
"$CXX" "${COMMON[@]}" apps/cli/main.mm services/turbociderd/service.mm -L"$OUT" -lturbocider -framework Foundation -Wl,-rpath,@executable_path -o "$OUT/turbocider"
printf 'Built %s\n' "$OUT/turbocider"
mkdir -p "$OUT/coreml"
cp tools/coreml/export_flux2.py "$OUT/coreml/"
tools/native/build_app.sh
