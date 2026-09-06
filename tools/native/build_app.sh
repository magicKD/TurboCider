#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
OUT="$PWD/build/native"
SDK="${SDKROOT:-$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk}"
SWIFTC="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc"
FLAGS=(-sdk "$SDK" -target arm64-apple-macosx26.2 -module-cache-path "$OUT/module-cache" -I bindings/c/include -L "$OUT" -lturbocider -Xlinker -rpath -Xlinker @executable_path -parse-as-library -O)
SDK_SOURCE=bindings/swift/TurboCiderNative.swift
STATE=(apps/macos/JobStore.swift apps/macos/StudioState.swift apps/macos/AccelerationDiscovery.swift)
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" apps/macos/MediaViews.swift apps/macos/AccelerationView.swift apps/macos/CoreMLStorageView.swift apps/macos/App.swift -o "$OUT/TurboCiderNativeApp"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/StudioBehaviorTests.swift -o "$OUT/turbocider-studio-tests"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/AppSmoke.swift -o "$OUT/turbocider-app-smoke"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/LifecycleTest.swift -o "$OUT/turbocider-lifecycle-test"
printf 'Built Swift App and integration tests\n'
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/StudioModelTests.swift -o "$OUT/turbocider-studio-model-tests"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" tests/integration/EndToEndBenchmark.swift -o "$OUT/turbocider-e2e-benchmark"
