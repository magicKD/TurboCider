#!/bin/sh
set -eu

PACKAGE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CONFIGURATION=${CONFIGURATION:-release}

cd "$PACKAGE_ROOT"
swift build -c "$CONFIGURATION"
BIN_DIR=$(swift build -c "$CONFIGURATION" --show-bin-path)
BUNDLE="$PACKAGE_ROOT/dist/TurboCider.app"
CONTENTS="$BUNDLE/Contents"
RESOURCES="$CONTENTS/Resources/TurboCider"

rm -rf "$BUNDLE"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES"
cp "$BIN_DIR/TurboCiderApp" "$CONTENTS/MacOS/TurboCiderApp"
cp "$BIN_DIR/turbocider-swift" "$CONTENTS/MacOS/turbocider-swift"
cp "$PACKAGE_ROOT/packaging/Info.plist" "$CONTENTS/Info.plist"
cp "$PACKAGE_ROOT/pyproject.toml" "$RESOURCES/pyproject.toml"
cp -R "$PACKAGE_ROOT/src" "$RESOURCES/src"
cp -R "$PACKAGE_ROOT/model-packs" "$RESOURCES/model-packs"
cp -R "$PACKAGE_ROOT/device-profiles" "$RESOURCES/device-profiles"
cp -R "$PACKAGE_ROOT/schemas" "$RESOURCES/schemas"
cp -R "$PACKAGE_ROOT/scripts" "$RESOURCES/scripts"
cp -R "$PACKAGE_ROOT/benchmarks" "$RESOURCES/benchmarks"
if [ -d "$PACKAGE_ROOT/Python" ]; then
    cp -R "$PACKAGE_ROOT/Python" "$RESOURCES/Python"
fi
if [ -d "$PACKAGE_ROOT/runtime" ]; then
    cp -R "$PACKAGE_ROOT/runtime" "$RESOURCES/runtime"
fi
if [ -d "$PACKAGE_ROOT/engines" ]; then
    cp -R "$PACKAGE_ROOT/engines" "$RESOURCES/engines"
fi

if command -v codesign >/dev/null 2>&1; then
    codesign --force --deep --sign - "$BUNDLE"
fi
"$CONTENTS/MacOS/TurboCiderApp" --smoke >/dev/null
echo "$BUNDLE"
