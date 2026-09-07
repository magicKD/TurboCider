#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
 if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
 else export DEVELOPER_DIR="$(xcode-select -p)"
 fi
fi
source tools/native/dependencies.sh
OUT="$PWD/build/native"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
TOOLCHAIN="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin"
[[ -d "$TOOLCHAIN" ]] || TOOLCHAIN="$DEVELOPER_DIR/usr/bin"
SWIFTC="$TOOLCHAIN/swiftc"
MLX_MIN_MACOS="$(otool -l "$MLX_ROOT/lib/libmlx.dylib" | awk '
 /cmd LC_BUILD_VERSION/{build=1; next}
 build && /minos /{print $2; exit}
')"
DEPLOYMENT_TARGET="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}"
FLAGS=(-sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -I bindings/c/include -L "$OUT" -lturbocider -Xlinker -rpath -Xlinker @executable_path -parse-as-library -O)
SDK_SOURCE=bindings/swift/TurboCiderNative.swift
STATE=(apps/macos/JobStore.swift apps/macos/StudioState.swift apps/macos/AccelerationDiscovery.swift apps/macos/ResourceMonitor.swift apps/macos/ResourceInventory.swift)
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" apps/macos/MediaViews.swift apps/macos/AccelerationView.swift apps/macos/CoreMLStorageView.swift apps/macos/App.swift -o "$OUT/TurboCiderNativeApp"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/StudioBehaviorTests.swift -o "$OUT/turbocider-studio-tests"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/AppSmoke.swift -o "$OUT/turbocider-app-smoke"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/LifecycleTest.swift -o "$OUT/turbocider-lifecycle-test"
printf 'Built Swift App and integration tests\n'
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/StudioModelTests.swift -o "$OUT/turbocider-studio-model-tests"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/EndToEndBenchmark.swift -o "$OUT/turbocider-e2e-benchmark"
