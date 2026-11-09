#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/native/dependencies.sh
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
 if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
 else export DEVELOPER_DIR="$(xcode-select -p)"; fi
fi
APP="${TURBOCIDER_DESKTOP_APP:-$PWD/dist/TurboCider.app}"
if [[ "$APP" != /* ]]; then APP="$PWD/$APP"; fi
OUT="${TURBOCIDER_WAN22_APP_TEST_OUT:-$PWD/build/wan22-desktop-e2e}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
BIN="$APP/Contents/MacOS"
test -f "$BIN/libturbocider.dylib"
mkdir -p "$OUT/module-cache"
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
TOOLCHAIN="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin"
[[ -d "$TOOLCHAIN" ]] || TOOLCHAIN="$DEVELOPER_DIR/usr/bin"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
FLAGS=(-sdk "$SDK" -target "arm64-apple-macosx${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}"
 -module-cache-path "$OUT/module-cache" -I bindings/c/include -L "$BIN" -lturbocider
 -Xlinker -rpath -Xlinker "$BIN" -parse-as-library -O -whole-module-optimization)
STATE=(apps/macos/ZImageInstallation.swift apps/macos/JobStore.swift apps/macos/ReferenceImagePreparation.swift
 apps/macos/StudioState.swift apps/macos/AccelerationDiscovery.swift apps/macos/ResourceMonitor.swift apps/macos/ResourceInventory.swift
 apps/macos/Qwen21Installation.swift services/model-library/LibraryStore.swift services/model-library/LibrarySettings.swift
 services/model-library/LibraryRecipes.swift services/model-library/HubClient.swift services/model-library/LibraryDownload.swift
 services/model-library/LibraryANE.swift services/model-library/LibraryInventory.swift services/model-library/TinyVAEAssets.swift
 services/model-library/TensorCache.swift services/model-library/DiagnosticTensorCache.swift services/model-library/InstallationInspection.swift
 services/model-library/LibraryComponents.swift apps/macos/ModelLibraryController.swift apps/macos/LocalAPIController.swift
 apps/macos/LocalHTTPGateway.swift apps/macos/RunInsights.swift apps/macos/TensorCacheController.swift apps/macos/LTXWorker.swift
 apps/macos/ImageOutputTransaction.swift apps/macos/WorkerRequestEnvelope.swift apps/macos/WorkerTerminalEnvelope.swift
 apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift apps/macos/PublicImageWorker.swift
 apps/macos/PublicImageQueries.swift apps/macos/WorkerEventStream.swift apps/macos/RuntimeImageWorker.swift apps/macos/VideoPreview.swift
 apps/macos/TinyVAE.swift apps/macos/SeedVR2ResourceAdvice.swift apps/macos/SeedVR2Workflow.swift apps/macos/HistorySelection.swift
 apps/macos/ImageUpscaler.swift apps/macos/ModelHubPresentation.swift apps/macos/EditingCanvasSizing.swift
 apps/macos/ItemProviderDataLoader.swift apps/macos/PlaygroundState.swift)
"$TOOLCHAIN/swiftc" "${FLAGS[@]}" bindings/swift/TurboCiderNative.swift "${STATE[@]}" \
 tests/integration/Wan22DesktopE2E.swift -o "$OUT/wan22-desktop-e2e"
