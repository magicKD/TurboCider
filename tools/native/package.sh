#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
ROOT="$PWD"
APP="$ROOT/dist/TurboCider.app"
BIN="$APP/Contents/MacOS"
RES="$APP/Contents/Resources"
mkdir -p "$BIN" "$RES" "$ROOT/dist/cli"
cp build/native/TurboCiderNativeApp "$BIN/"
for folder in "$BIN" "$ROOT/dist/cli"; do
 mkdir -p "$folder/coreml"
 cp tools/coreml/export_flux2.py "$folder/coreml/"
 cp build/native/libturbocider.dylib "$folder/"
 cp "$MLX_ROOT/lib/libmlx.dylib" "$MLX_ROOT/lib/libjaccl.dylib" "$MLX_ROOT/lib/mlx.metallib" "$folder/"
 install_name_tool -delete_rpath "$MLX_ROOT/lib" "$folder/libturbocider.dylib"
 install_name_tool -add_rpath @loader_path "$folder/libturbocider.dylib"
 # Third-party binaries may carry build-machine rpaths; remove every such path.
 for library in "$folder/libmlx.dylib" "$folder/libjaccl.dylib"; do
  while IFS= read -r rpath; do
   [ -n "$rpath" ] && install_name_tool -delete_rpath "$rpath" "$library"
  done < <(otool -l "$library" | awk '/cmd LC_RPATH/{found=1; next} found && /path /{print $2;found=0}')
  install_name_tool -add_rpath @loader_path "$library"
 done
 for library in "$folder"/*.dylib; do codesign --force --sign - "$library"; done
done
cp build/native/turbocider "$ROOT/dist/cli/"
cp native/THIRD_PARTY_NOTICES.md "$RES/"
cp "$MLX_ROOT/../mlx-0.32.0.dist-info/licenses/LICENSE" "$RES/MLX-LICENSE.txt"
cp "$RES/THIRD_PARTY_NOTICES.md" "$RES/MLX-LICENSE.txt" "$ROOT/dist/cli/"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>org.turbocider.native</string>
<key>CFBundleName</key><string>TurboCider</string>
<key>CFBundleExecutable</key><string>TurboCiderNativeApp</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.2.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>26.2</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$ROOT/dist/cli/turbocider"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
printf 'Local ad-hoc signed App: %s\n' "$APP"
