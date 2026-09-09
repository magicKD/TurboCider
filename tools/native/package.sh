#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
ROOT="$PWD"
APP="$ROOT/dist/TurboCider.app"
BIN="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"
MLX_LICENSE_PATH="${MLX_LICENSE_PATH:-}"
if [[ -z "$MLX_LICENSE_PATH" ]]; then
 shopt -s nullglob
 mlx_license_candidates=("$MLX_ROOT"/../mlx-*.dist-info/licenses/LICENSE)
 shopt -u nullglob
 [[ ${#mlx_license_candidates[@]} -eq 1 ]] || {
  echo "Cannot identify one MLX license; set MLX_LICENSE_PATH" >&2
  exit 1
 }
 MLX_LICENSE_PATH="${mlx_license_candidates[0]}"
fi
[[ -f "$MLX_LICENSE_PATH" ]] || {
 echo "MLX license is missing: $MLX_LICENSE_PATH" >&2
 exit 1
}
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
PACKAGE_MIN_MACOS="${TURBOCIDER_PACKAGE_MIN_MACOS:-${MLX_MIN_MACOS:-15.0}}"
rm -rf "$APP" "$ROOT/dist/cli"
mkdir -p "$BIN" "$RES" "$ROOT/dist/cli"
cp build/native/TurboCiderNativeApp "$BIN/"
cp build/native/turbocider "$BIN/"
for folder in "$BIN" "$ROOT/dist/cli"; do
 cp build/native/libturbocider.dylib "$folder/"
 cp build/native/h3-quantize-stream-cache "$folder/"
 cp "$MLX_ROOT/lib/libmlx.dylib" "$MLX_ROOT/lib/libjaccl.dylib" "$MLX_ROOT/lib/mlx.metallib" "$folder/"
 cp build/native/h3_shaders.metal build/native/ltx_shaders.metal "$folder/"
 cp build/native/ltx-video-finalizer build/native/ltx-video-vae-decode "$folder/"
 install_name_tool -delete_rpath "$MLX_ROOT/lib" "$folder/libturbocider.dylib"
 install_name_tool -add_rpath @loader_path "$folder/libturbocider.dylib"
 # Third-party binaries may carry build-machine rpaths; remove every such path.
 for library in "$folder/libmlx.dylib" "$folder/libjaccl.dylib"; do
  while IFS= read -r rpath; do
   [ -n "$rpath" ] && install_name_tool -delete_rpath "$rpath" "$library"
  done < <(otool -l "$library" | awk '/cmd LC_RPATH/{found=1; next} found && /path /{print $2;found=0}')
  install_name_tool -add_rpath @loader_path "$library"
 done
 for helper in "$folder/ltx-video-finalizer" "$folder/ltx-video-vae-decode" \
               "$folder/h3-quantize-stream-cache"; do
  while IFS= read -r rpath; do
   [ -n "$rpath" ] && install_name_tool -delete_rpath "$rpath" "$helper"
  done < <(otool -l "$helper" | awk '/cmd LC_RPATH/{found=1; next} found && /path /{print $2;found=0}')
  install_name_tool -add_rpath @loader_path "$helper"
  codesign --force --sign - "$helper"
 done
 for library in "$folder"/*.dylib; do codesign --force --sign - "$library"; done
done
cp build/native/turbocider "$ROOT/dist/cli/"
# LoRA conversion/merge scripts are release-pipeline tools only.  They are
# intentionally not copied into the App bundle: production sessions accept
# provenance-verified premerged checkpoints and never launch Python.
cp native/THIRD_PARTY_NOTICES.md "$RES/"
cp "$MLX_LICENSE_PATH" "$RES/MLX-LICENSE.txt"
cp "$RES/THIRD_PARTY_NOTICES.md" "$RES/MLX-LICENSE.txt" "$ROOT/dist/cli/"
cp native/licenses/FastVideo-LICENSE.txt native/licenses/TAEHV-LICENSE.txt "$RES/"
cp native/licenses/FastVideo-LICENSE.txt native/licenses/TAEHV-LICENSE.txt "$ROOT/dist/cli/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>org.turbocider.native</string>
<key>CFBundleName</key><string>TurboCider</string>
<key>CFBundleExecutable</key><string>TurboCiderNativeApp</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.2.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>${PACKAGE_MIN_MACOS}</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$ROOT/dist/cli/turbocider"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
printf 'Local ad-hoc signed App: %s\n' "$APP"
printf 'Packaged minimum macOS: %s\n' "$PACKAGE_MIN_MACOS"
