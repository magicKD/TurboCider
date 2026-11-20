#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -z "${DEVELOPER_DIR:-}" ]]; then
 if [[ -d /Applications/Xcode.app/Contents/Developer ]]; then export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
 else export DEVELOPER_DIR="$(xcode-select -p)"
 fi
fi
source tools/native/dependencies.sh
APP_ONLY="${TURBOCIDER_BUILD_APP_ONLY:-0}"
case "$APP_ONLY" in
 0|1) ;;
 *) printf 'TURBOCIDER_BUILD_APP_ONLY must be 0 or 1\n' >&2; exit 2 ;;
esac
OUT="${TURBOCIDER_BUILD_OUTPUT_DIR:-${TURBOCIDER_NATIVE_OUT:-$PWD/build/native}}"
if [[ "$OUT" != /* ]]; then OUT="$PWD/$OUT"; fi
SDK="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
TOOLCHAIN="$DEVELOPER_DIR/Toolchains/XcodeDefault.xctoolchain/usr/bin"
[[ -d "$TOOLCHAIN" ]] || TOOLCHAIN="$DEVELOPER_DIR/usr/bin"
SWIFTC="$TOOLCHAIN/swiftc"
MLX_MIN_MACOS="$(turbocider_mlx_min_macos)"
DEPLOYMENT_TARGET="${TURBOCIDER_DEPLOYMENT_TARGET:-${MLX_MIN_MACOS:-15.0}}"
FLAGS=(-sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -I bindings/c/include -L "$OUT" -lturbocider -Xlinker -rpath -Xlinker @executable_path -parse-as-library -O -whole-module-optimization)
SDK_SOURCE=bindings/swift/TurboCiderNative.swift
STATE=(apps/macos/ZImageInstallation.swift apps/macos/JobStore.swift apps/macos/ReferenceImagePreparation.swift apps/macos/StudioState.swift apps/macos/AccelerationDiscovery.swift apps/macos/ResourceMonitor.swift apps/macos/ResourceInventory.swift)
STATE+=(apps/macos/Qwen21Installation.swift)
LIBRARY=(services/model-library/LibraryStore.swift services/model-library/LibrarySettings.swift services/model-library/LibraryRecipes.swift services/model-library/HubClient.swift services/model-library/LibraryDownload.swift)
LIBRARY+=(services/model-library/LibraryANE.swift)
LIBRARY+=(services/model-library/LibraryInventory.swift)
LIBRARY+=(services/model-library/TinyVAEAssets.swift)
LIBRARY+=(services/model-library/TensorCache.swift)
LIBRARY+=(services/model-library/DiagnosticTensorCache.swift)
LIBRARY+=(services/model-library/InstallationInspection.swift services/model-library/LibraryComponents.swift)
STATE+=("${LIBRARY[@]}" apps/macos/ModelLibraryController.swift apps/macos/LocalAPIController.swift apps/macos/LocalHTTPGateway.swift)
STATE+=(apps/macos/RunInsights.swift)
STATE+=(apps/macos/TensorCacheController.swift)
STATE+=(apps/macos/LTXWorker.swift)
STATE+=(apps/macos/ImageOutputTransaction.swift)
STATE+=(apps/macos/WorkerRequestEnvelope.swift apps/macos/WorkerTerminalEnvelope.swift apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift apps/macos/PublicImageWorker.swift apps/macos/PublicImageQueries.swift apps/macos/WorkerEventFraming.swift apps/macos/WorkerEventStream.swift apps/macos/RuntimeImageWorker.swift)
STATE+=(apps/macos/VideoPreview.swift)
STATE+=(apps/macos/TinyVAE.swift)
STATE+=(apps/macos/SeedVR2ResourceAdvice.swift)
STATE+=(apps/macos/SeedVR2Workflow.swift)
STATE+=(apps/macos/HistorySelection.swift apps/macos/ImageUpscaler.swift)
STATE+=(apps/macos/ModelHubPresentation.swift)
STATE+=(apps/macos/EditingCanvasSizing.swift apps/macos/ItemProviderDataLoader.swift apps/macos/PlaygroundState.swift)
APP_VIEWS=(
 apps/macos/SeedVR2UpscaleControls.swift apps/macos/ImageUpscaleView.swift
 apps/macos/EditingCanvasSizingView.swift apps/macos/PlaygroundView.swift
 apps/macos/ReferenceImagePreparationView.swift apps/macos/ModelLibraryView.swift
 apps/macos/ModelLoRALibraryView.swift apps/macos/ANELibraryView.swift
 apps/macos/ModelDownloadView.swift apps/macos/LocalAPIView.swift
 apps/macos/RunInsightsView.swift apps/macos/TensorCacheView.swift
 apps/macos/MediaViews.swift apps/macos/AccelerationView.swift apps/macos/CoreMLStorageView.swift
 apps/macos/TinyVAEViews.swift apps/macos/TinyVAELibraryView.swift apps/macos/App.swift
)
build_app_integration_test() {
 local executable="$1"
 shift
 "$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" "$@" -o "$OUT/$executable"
}
APP_TARGETS="${TURBOCIDER_APP_TARGETS:-all}"
case "$APP_TARGETS" in
 all|seedvr2|upscale-presentation|tiny-vae|prunavaed|desktop-media) ;;
 *) printf 'TURBOCIDER_APP_TARGETS must be all, seedvr2, upscale-presentation, tiny-vae, prunavaed or desktop-media\n' >&2; exit 2 ;;
esac
if [[ "$APP_TARGETS" == prunavaed ]]; then
 build_app_integration_test turbocider-prunavaed-app-tests tests/integration/PrunaVAEAppTests.swift
 "$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/PrunaVAEDownloadTests.swift -o "$OUT/turbocider-prunavaed-download-tests"
 "$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/TinyVAEDownloadTests.swift -o "$OUT/turbocider-tiny-vae-download-tests"
 printf 'Built PrunaVAED App contract and shared VAE downloader tests\n'
 exit 0
fi
if [[ "$APP_TARGETS" == tiny-vae ]]; then
 "$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/TinyVAEDownloadTests.swift -o "$OUT/turbocider-tiny-vae-download-tests"
 build_app_integration_test turbocider-tiny-vae-app-tests tests/integration/TinyVAEAppTests.swift
 printf 'Built tiny VAE App contract and GPU acceptance tests\n'
 exit 0
fi
if [[ "$APP_TARGETS" == upscale-presentation ]]; then
 build_app_integration_test turbocider-upscale-presentation-tests apps/macos/SeedVR2UpscaleControls.swift tests/integration/UpscalePresentationTests.swift
 "$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/RunInsights.swift tests/integration/RunInsightsTests.swift -o "$OUT/turbocider-run-insights-tests"
 printf 'Built upscale presentation and run-insights tests\n'
 exit 0
fi
if [[ "$APP_ONLY" == 0 && "$APP_TARGETS" == all ]]; then
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/PrunaVAEDownloadTests.swift -o "$OUT/turbocider-prunavaed-download-tests"
build_app_integration_test turbocider-prunavaed-app-tests tests/integration/PrunaVAEAppTests.swift
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/TinyVAEDownloadTests.swift -o "$OUT/turbocider-tiny-vae-download-tests"
build_app_integration_test turbocider-tiny-vae-app-tests tests/integration/TinyVAEAppTests.swift
build_app_integration_test turbocider-playground-tests tests/integration/PlaygroundStateTests.swift
build_app_integration_test turbocider-editing-canvas-tests tests/integration/EditingCanvasSizingTests.swift
build_app_integration_test turbocider-reference-preparation-tests tests/integration/ReferenceImagePreparationTests.swift
build_app_integration_test turbocider-qwen21-app-tests tests/integration/Qwen21AppTests.swift
build_app_integration_test turbocider-qwen21-workflow-tests tests/integration/Qwen21WorkflowTests.swift
build_app_integration_test turbocider-qwen21-gpu-acceptance tests/integration/Qwen21GPUAcceptance.swift
fi
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" services/model-library/main.swift -o "$OUT/turbocider-library"
if [[ "$APP_ONLY" == 0 && "$APP_TARGETS" == all ]]; then
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/LibraryStoreTests.swift -o "$OUT/turbocider-library-store-tests"
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/HubClientTests.swift -o "$OUT/turbocider-hub-tests"
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/TensorCacheTests.swift -o "$OUT/turbocider-tensor-cache-tests"
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/LibraryComponentTests.swift -o "$OUT/turbocider-component-tests"
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O "${LIBRARY[@]}" tests/integration/InstallationInspectionTests.swift -o "$OUT/turbocider-installation-tests"
fi
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" "${STATE[@]}" "${APP_VIEWS[@]}" -o "$OUT/TurboCiderNativeApp"
if [[ "$APP_ONLY" == 1 ]]; then
 printf 'Built Swift App and model-library helper; test targets skipped\n'
 exit 0
fi
if [[ "$APP_TARGETS" == desktop-media ]]; then
 build_app_integration_test turbocider-desktop-media-tests tests/integration/DesktopMediaWorkflowTests.swift
 build_app_integration_test turbocider-keyframe-workflow-tests tests/integration/KeyframeWorkflowTests.swift
 "$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O apps/macos/VideoPreview.swift tests/integration/VideoPreviewTests.swift -o "$OUT/turbocider-video-preview-tests"
 printf 'Built desktop media workflow, video preview tests and App\n'
 exit 0
fi
if [[ "$APP_TARGETS" == seedvr2 ]]; then
 build_app_integration_test turbocider-seedvr2-app-tests tests/integration/SeedVR2AppTests.swift
 build_app_integration_test turbocider-model-hub-tests tests/integration/ModelHubPresentationTests.swift
 printf 'Built Swift App, model-library helper, SeedVR2 and model-hub integration tests\n'
 exit 0
fi
build_app_integration_test turbocider-studio-tests tests/integration/StudioBehaviorTests.swift tests/integration/StudioStreamingQueryTests.swift
build_app_integration_test turbocider-upscaler-tests tests/integration/ImageUpscalerTests.swift
build_app_integration_test turbocider-history-tests tests/integration/HistoryManagementTests.swift tests/integration/PublicImageJobTests.swift
build_app_integration_test turbocider-desktop-media-tests tests/integration/DesktopMediaWorkflowTests.swift
build_app_integration_test turbocider-keyframe-workflow-tests tests/integration/KeyframeWorkflowTests.swift
build_app_integration_test turbocider-app-smoke tests/integration/AppSmoke.swift
build_app_integration_test turbocider-lifecycle-test tests/integration/LifecycleTest.swift
build_app_integration_test turbocider-studio-model-tests tests/integration/StudioModelTests.swift
build_app_integration_test turbocider-seedvr2-app-tests tests/integration/SeedVR2AppTests.swift
build_app_integration_test turbocider-e2e-benchmark tests/integration/EndToEndBenchmark.swift
build_app_integration_test turbocider-z-image-app-tests tests/integration/ZImageAppTests.swift
build_app_integration_test turbocider-studio-controls-tests tests/integration/StudioControlsTests.swift
build_app_integration_test turbocider-z-image-shape-tests tests/integration/ZImageShapeTests.swift
build_app_integration_test turbocider-z-image-partition-routing-tests tests/integration/ZImagePartitionRoutingTests.swift
build_app_integration_test turbocider-model-hub-tests tests/integration/ModelHubPresentationTests.swift
build_app_integration_test turbocider-lora-management-tests tests/integration/ModelLoRAManagementTests.swift
build_app_integration_test turbocider-model-library-tests tests/integration/ModelLibraryTests.swift
build_app_integration_test turbocider-library-tool-tests tests/integration/LibraryToolTests.swift
build_app_integration_test turbocider-local-api-tests tests/integration/LocalAPITests.swift
"$SWIFTC" "${FLAGS[@]}" apps/macos/LocalHTTPGateway.swift tests/integration/LocalHTTPGatewayTests.swift -o "$OUT/turbocider-http-gateway-tests"
build_app_integration_test turbocider-run-insights-tests tests/integration/RunInsightsTests.swift
build_app_integration_test turbocider-studio-gpu-model-tests tests/integration/StudioGPUModelTests.swift

build_app_integration_test turbocider-ane-library-tests tests/integration/ANELibraryTests.swift
build_app_integration_test turbocider-studio-variant-tests tests/integration/StudioVariantTests.swift
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" tests/integration/ZImagePromptTests.swift -o "$OUT/turbocider-z-image-prompt-tests"
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library -O apps/macos/VideoPreview.swift tests/integration/VideoPreviewTests.swift -o "$OUT/turbocider-video-preview-tests"

"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" tests/integration/StreamingResolutionTests.swift -o "$OUT/turbocider-streaming-resolution-tests"
build_app_integration_test turbocider-ltx-worker-tests tests/integration/LTXWorkerTests.swift
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/ImageOutputTransaction.swift tests/integration/ImageOutputTransactionTests.swift -o "$OUT/turbocider-image-transaction-tests"

"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/WorkerRequestEnvelope.swift apps/macos/WorkerTerminalEnvelope.swift tests/integration/WorkerTerminalEnvelopeTests.swift -o "$OUT/turbocider-worker-terminal-tests"

"$TOOLCHAIN/clang" -isysroot "$SDK" -mmacosx-version-min="$DEPLOYMENT_TARGET" -Wall -Wextra -Werror tests/integration/native_process_fixture.c -o "$OUT/turbocider-process-fixture"
"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift tests/integration/NativeProcessRunnerTests.swift -o "$OUT/turbocider-process-runner-tests"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/WorkerRequestEnvelope.swift apps/macos/WorkerTerminalEnvelope.swift apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift tests/integration/NativeProcessWorkerTests.swift -o "$OUT/turbocider-process-worker-tests"

"$SWIFTC" -sdk "$SDK" -target "arm64-apple-macosx${DEPLOYMENT_TARGET}" -module-cache-path "$OUT/module-cache" -parse-as-library apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift tests/integration/WorkerLaunchAdmissionTests.swift -o "$OUT/turbocider-worker-admission-tests"

build_app_integration_test turbocider-public-image-app-tests tests/integration/PublicImageAppModelTests.swift
build_app_integration_test turbocider-public-streaming-smoke tests/integration/PublicImageStreamingSmoke.swift

"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/WorkerRequestEnvelope.swift apps/macos/WorkerTerminalEnvelope.swift apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift apps/macos/PublicImageWorker.swift apps/macos/PublicImageQueries.swift tests/integration/PublicImageQueryTests.swift -o "$OUT/turbocider-public-image-query-tests"

"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/ImageUpscaler.swift tests/integration/UpscaleComputeBenchmark.swift -o "$OUT/turbocider-upscale-benchmark"
"$SWIFTC" "${FLAGS[@]}" "$SDK_SOURCE" apps/macos/WorkerRequestEnvelope.swift apps/macos/WorkerTerminalEnvelope.swift apps/macos/WorkerProcessIdentity.swift apps/macos/NativeProcessRunner.swift apps/macos/PublicImageWorker.swift apps/macos/WorkerEventFraming.swift apps/macos/RuntimeImageWorker.swift tests/integration/RuntimeImageWorkerTests.swift -o "$OUT/turbocider-runtime-image-worker-tests"
printf 'Built Swift App and integration tests\n'
