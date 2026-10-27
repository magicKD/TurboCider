import AppKit
import Foundation

@main struct SeedVR2AppTests {
    @MainActor static func main() async throws {
        guard [3, 5, 6, 7].contains(CommandLine.arguments.count) else {
            throw NativeFailure(message: "seedvr2-app-tests SOURCE_VIDEO OUTPUT_DIRECTORY [MODEL_ROOT SOURCE_IMAGE [resident|component_staged [ANE_MANIFEST]]]")
        }
        let residency = CommandLine.arguments.count >= 6 ? CommandLine.arguments[5] : "resident"
        let manifest = CommandLine.arguments.count == 7 ? CommandLine.arguments[6] : nil
        let hybrid = manifest != nil
        let execution = hybrid ? "gpu_ane" : "gpu"
        let causalManifest = ProcessInfo.processInfo.environment["TURBOCIDER_SEEDVR2_VAE_CAUSAL_MANIFEST"]
        let causalExperiment = causalManifest?.isEmpty == false
        let upscaleCausalManifest = ProcessInfo.processInfo.environment["TURBOCIDER_SEEDVR2_VAE_UPSCALE_CAUSAL_MANIFEST"]
        let upscaleCausalExperiment = upscaleCausalManifest?.isEmpty == false
        let vaeOptions = try ProcessInfo.processInfo.environment["TURBOCIDER_SEEDVR2_APP_VAE_OPTIONS"].map {
            try JSONDecoder().decode(NativeSeedVR2VAEConfig.self, from: Data($0.utf8))
        }
        guard ["resident", "component_staged"].contains(residency) else {
            throw NativeFailure(message: "unsupported SeedVR2 test residency")
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[2])
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let studio = StudioState(directory: root)
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw NativeFailure(message: message) }
        }
        let model = try StudioModel.catalog().first(where: { $0.id == "seedvr2-3b" }).unwrap("SeedVR2 catalog entry missing")
        try check(model.supports("image.upscale") && model.supports("video.upscale"), "both upscale operations must be advertised")
        let quick = StudioState(directory: root.appendingPathComponent("quick-controls"))
        let oldID = quick.draft.modelID
        try check(!quick.activateSeedVR2Upscaling(kind: "unsupported") && quick.draft.modelID == oldID,
                  "invalid SeedVR2 workflow mutated the generation draft")
        quick.importing = true
        try check(!quick.activateSeedVR2Upscaling(kind: "image") && quick.draft.modelID == oldID,
                  "SeedVR2 workflow changed during an input import")
        quick.importing = false
        try check(quick.activateSeedVR2Upscaling(kind: "image") && quick.draft.modelID == model.id &&
                  quick.draft.operation == "image.upscale" && quick.draft.steps == 1 && !quick.draft.audio &&
                  quick.draft.acceleration?.policy == "gpu", "SeedVR2 quick entry did not select safe GPU defaults")
        let rectangle = StudioAsset(path: "metadata-only.png", name: "rectangle", width: 320, height: 160)
        quick.draft.assets = [rectangle]; quick.draft.initImageID = rectangle.id
        quick.applySeedVR2OutputPreset(.long512)
        try check(quick.draft.width == 512 && quick.draft.height == 256, "long-edge preset distorted a rectangle")
        quick.applySeedVR2OutputPreset(.scale4)
        try check(quick.draft.width == 1280 && quick.draft.height == 640, "4x source preset dimensions differ")
        var tooLarge = rectangle; tooLarge.width = 2000; tooLarge.height = 1000
        quick.draft.assets = [tooLarge]
        quick.applySeedVR2OutputPreset(.scale2)
        try check(quick.message != nil && quick.draft.width == 1280 && quick.draft.height == 640,
                  "oversized preset silently clamped output or partially changed the draft")
        var malformed = rectangle; malformed.width = Int.max
        do { _ = try SeedVR2OutputPreset.scale4.dimensions(source: malformed); throw NativeFailure(message: "malformed source accepted") }
        catch let error as NativeFailure { try check(error.message.contains("源媒体尺寸"), "wrong malformed-source rejection") }
        var clip = rectangle; clip.kind = "video"; clip.frames = 25; clip.fps = 24
        quick.draft.assets = [clip]
        try check(quick.activateSeedVR2Upscaling(kind: "video") && quick.draft.frames == 25 && quick.draft.fps == 24,
                  "SeedVR2 video quick entry lost source frame/FPS binding")
        quick.applySeedVR2OutputPreset(.long1024)
        try check(quick.draft.width == 1024 && quick.draft.height == 512 && quick.draft.frames == 25 && quick.draft.fps == 24,
                  "resolution shortcut truncated or retimed the source video")
        quick.applySeedVR2MemoryAdvice()
        quick.applySeedVR2SpeedSettings()
        try check(quick.draft.residency == "resident" && quick.draft.seedvr2VAE?.history == "frames" &&
                  quick.draft.seedvr2VAE?.allocator_cache_bytes == UInt64(4)<<30 &&
                  quick.draft.seedvr2VAE?.spatial == "full" && quick.draft.frames == 25 && quick.draft.fps == 24 &&
                  quick.draft.width == 1024 && quick.draft.height == 512,"speed preference changed media or did not reset memory-only schedules")
        studio.draft.modelPaths[model.id] = root.path
        studio.selectModel(model.id)
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 512, pixelsHigh: 512,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        memset(bitmap.bitmapData!, 255, bitmap.bytesPerRow * bitmap.pixelsHigh)
        let sourceImage = root.appendingPathComponent("source.png")
        try bitmap.representation(using: .png, properties: [:])!.write(to: sourceImage, options: .atomic)
        await studio.addFiles([sourceImage])
        studio.draft.prompt = "This old text must not become SeedVR2 conditioning"
        let image = try studio.draft.request(output: root.appendingPathComponent("image.png"))
        try check(image.operation == "image.upscale" && image.prompt.isEmpty && image.steps == 1 && image.frames == 1,
                  "image request must use the embedded conditioning and one-step restoration")
        try check(image.inputs?.first?.role == "source" && image.inputs?.first?.kind == "image", "image source role/kind mismatch")
        let imageV2 = NativeRequestV2(legacy: image)
        try check(imageV2.outputs[0].kind == "image", "schema-2 image output kind mismatch")
        try check(UpscaleModelChoice.allCases.filter(\.supportsVideo) == [.seedvr2],"image-only model advertised for video")
        let frameProgress = SeedVR2ProgressPresentation(phase: "seedvr2_decode_progress",completed: 49,total: 97)
        try check(frameProgress.stage == 3 && frameProgress.detail.contains("49 / 97") && frameProgress.fraction! > 0.5,
                  "whole-video decode progress unclear")
        try check(frameProgress.remainingStageSeconds(elapsed: 98) == 96,"stage ETA differs")
        try check(SeedVR2ProgressPresentation(phase: "seedvr2_dit_stage_load",completed: 0,total: 1).fraction == nil,
                  "model reload presented as completed frames")
        let imageHistory = NativeJob(id: UUID(), createdAt: Date(), request: image,
            state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1,
            modelPath: root.path)
        let historyStudio = StudioState(directory: root.appendingPathComponent("history-reuse"))
        historyStudio.reuse(imageHistory)
        try check(historyStudio.draft.modelID == model.id && historyStudio.draft.operation == "image.upscale" &&
                  historyStudio.draft.activeAssets.first?.mediaKind == "image",
                  "SeedVR2 history was mistaken for the auxiliary Real-ESRGAN upscaler")
        try check(imageHistory.routeSummary == nil,
                  "SeedVR2 history fabricated an auxiliary Core ML x4 execution label")
        for policy in ["resident", "component_staged"] {
            studio.draft.residency = policy
            let intent = try studio.draft.request(output: root.appendingPathComponent("residency-\(policy).png"))
            try check(intent.residency == policy, "draft lost the chosen SeedVR2 residency")
            try check(NativeRequestV2(legacy: intent).execution.residency == policy, "schema-2 lost the chosen residency")
        }
        studio.draft.residency = "resident"
        studio.changeOperation("video.upscale")
        try check(studio.creationKind == "video" && studio.supportsVideoInputs, "video workspace/import is unavailable")
        await studio.addFiles([URL(fileURLWithPath: CommandLine.arguments[1])])
        try check(studio.message == nil && studio.draft.activeAssets.count == 1, "real video staging failed")
        let video = try studio.draft.request(output: root.appendingPathComponent("video.mp4"))
        let config = NativeSeedVR2VAEConfig(decoder: "streaming", convolution: "mixed3d", tileRows: 4096, allocatorCacheBytes: 1 << 30, encoder: "streaming")
        studio.draft.seedvr2VAE = config
        let configured = try studio.draft.request(output: root.appendingPathComponent("configured.mp4"))
        try check(configured.seedvr2_vae == config && NativeRequestV2(legacy: configured).execution.seedvr2_vae == config,
                  "App/SDK lost request-local VAE controls")
        let restoredOptions = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(studio.draft))
        try check(restoredOptions.seedvr2VAE == config, "App draft did not persist VAE controls")
        let legacyVAE = try JSONDecoder().decode(NativeSeedVR2VAEConfig.self,from: Data("{\"decoder\":\"streaming\"}".utf8))
        try check(legacyVAE.encoder == "buffered" && legacyVAE.spatial == "full" && legacyVAE.encoder_spatial == "full" && legacyVAE.spatial_rows == 64,
            "legacy VAE options did not retain buffered input/full spatial defaults")
        try check(!legacyVAE.share_upscale_history,"legacy VAE unexpectedly enabled upscale sharing")
        let sharing = NativeSeedVR2VAEConfig(history: "frames", shareUpscaleHistory: true)
        studio.draft.seedvr2VAE = sharing
        let sharingRequest = try studio.draft.request(output: root.appendingPathComponent("sharing.mp4"))
        try check(sharingRequest.seedvr2_vae == sharing && NativeRequestV2(legacy: sharingRequest).execution.seedvr2_vae == sharing,
                  "App/SDK dropped upscale history sharing")
        try check(try JSONDecoder().decode(StudioDraft.self,from: JSONEncoder().encode(studio.draft)).seedvr2VAE == sharing,
                  "App draft did not persist upscale history sharing")
        let adviceSnapshot = studio.draft
        studio.draft.seedvr2VAE = nil
        studio.applySeedVR2MemoryAdvice()
        guard let advised = studio.draft.seedvr2VAE else { throw NativeFailure(message: "memory advice omitted VAE options") }
        try check(studio.draft.residency == "component_staged" && advised.encoder == "streaming" && advised.decoder == "streaming" &&
                  advised.history == "frames" && advised.share_upscale_history && advised.allocator_cache_bytes == (UInt64(1) << 30),
                  "memory advice did not apply supported streaming/sharing/cache settings")
        var explicitRows = advised; explicitRows.spatial = "file_rows"; explicitRows.share_upscale_history = false
        studio.draft.seedvr2VAE = explicitRows
        studio.applySeedVR2MemoryAdvice()
        try check(studio.draft.seedvr2VAE?.spatial == "file_rows" && studio.draft.seedvr2VAE?.share_upscale_history == false,
                  "memory advice changed explicit rows or selected incompatible sharing")
        studio.changeOperation("image.upscale");studio.applySeedVR2MemoryAdvice()
        try check(studio.draft.seedvr2VAE?.encoder == "buffered" && studio.draft.seedvr2VAE?.decoder == "buffered" &&
                  studio.draft.seedvr2VAE?.share_upscale_history == false,"image advice selected invalid streaming or unused sharing")
        func resourceAdvice(_ options: NativeSeedVR2VAEConfig, free: UInt64?, sourceFrames: Int? = 25,
                            configuredFrames: Int = 25, width: Int = 1024, height: Int = 1024,
                            physical: UInt64 = UInt64(64) << 30, operation: String = "video.upscale") -> [SeedVR2ResourceAdvice.Notice] {
            SeedVR2ResourceAdvice.notices(operation: operation,width: width,height: height,sourceFrames: sourceFrames,
                configuredFrames: configuredFrames,physicalMemory: physical,options: options,temporaryFreeBytes: free)
        }
        var rows = NativeSeedVR2VAEConfig();rows.spatial = "file_rows"
        let lowSpace = resourceAdvice(rows,free: UInt64(6) << 30)
        try check(lowSpace.contains { $0.id == "scratch_space_low" } && lowSpace.contains { $0.id == "scratch_reference" },
            "known 1024 row-file disk pressure was not explained")
        var combined = rows;combined.encoder_spatial = "file_rows"
        try check(!resourceAdvice(combined,free: UInt64(15) << 29).contains { $0.id == "scratch_space_low" },
            "independent encoder/decoder file peaks were incorrectly summed")
        var encoderRows = NativeSeedVR2VAEConfig();encoderRows.encoder_spatial = "file_rows"
        try check(!resourceAdvice(encoderRows,free: UInt64(3) << 30).contains { $0.id == "scratch_space_low" },
            "encoder-only advice incorrectly used the decoder scratch peak")
        rows.spatial_disk_budget_bytes = UInt64(4) << 30
        try check(resourceAdvice(rows,free: UInt64(10) << 30).contains { $0.id == "scratch_quota_low" },
            "explicit scratch quota was ignored")
        let missingSpace = resourceAdvice(rows,free: nil)
        try check(missingSpace.contains { $0.id == "scratch_space_unknown" } && !missingSpace.contains { $0.id == "scratch_space_low" },
            "unavailable free-space observation was converted into zero")
        try check(resourceAdvice(rows,free: UInt64(10) << 30,width: 768).contains { $0.id == "scratch_reference_unknown" },
            "an unmeasured rectangular target inherited a square scratch guarantee")
        let imageAdvice = resourceAdvice(rows,free: 0,sourceFrames: nil,configuredFrames: 1,operation: "image.upscale")
        try check(imageAdvice.contains { $0.id == "image_no_history_files" } && !imageAdvice.contains { $0.id == "scratch_space_low" },
            "single-frame advice incorrectly required video history files")
        let longSource = resourceAdvice(rows,free: UInt64(10) << 30,sourceFrames: 200,configuredFrames: 25)
        try check(longSource.contains { $0.id == "model_frame_limit" } && longSource.contains { $0.id == "source_frame_mismatch" },
            "imported full source frame count was replaced by a shorter draft")
        try check(resourceAdvice(rows,free: UInt64(10) << 30,physical: UInt64(16) << 30).contains { $0.id == "low_memory_unqualified" },
            "low-memory advice implied qualified 16-GiB operation")
        studio.draft = adviceSnapshot
        studio.draft.seedvr2VAE = nil
        try check(video.operation == "video.upscale" && video.frames == 25 && video.fps == 24,
                  "source frame count/FPS were not applied to the draft")
        try check(video.inputs?.first?.kind == "video" && video.inputs?.first?.role == "source", "video source role/kind mismatch")
        try check(NativeRequestV2(legacy: video).outputs[0].kind == "video", "schema-2 video output kind mismatch")
        let videoHistory = NativeJob(id: UUID(), createdAt: Date(), request: video,
            state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1,
            modelPath: root.path)
        historyStudio.reuse(videoHistory)
        try check(historyStudio.draft.activeAssets.first?.mediaKind == "video" &&
                  historyStudio.draft.activeAssets.first?.frames == video.frames &&
                  historyStudio.draft.activeAssets.first?.fps == video.fps,
                  "old/API video history lost its source kind/frame/FPS metadata during reuse")
        let source = studio.draft.activeAssets[0]
        try check(FileManager.default.fileExists(atPath: source.path) && source.width == 512 && source.height == 512,
                  "video source was not staged with oriented dimensions")
        var alternate = source
        alternate.id = UUID(); alternate.frames = 17; alternate.fps = 30
        studio.draft.assets.append(alternate)
        studio.selectSourceAsset(alternate.id)
        try check(studio.draft.frames == 17 && studio.draft.fps == 30, "changing source did not refresh frame/FPS binding")
        studio.selectSourceAsset(source.id)
        studio.draft.assets.removeAll { $0.id == alternate.id }
        try check(studio.draft.frames == 25 && studio.draft.fps == 24, "original source frame/FPS binding was not restored")
        var imageOnly = studio.draft
        imageOnly.operation = "image.edit"
        try check(imageOnly.activeAssets.allSatisfy { $0.mediaKind == "image" }, "retained video leaked into an image-only operation")
        studio.draft.frames = 24
        do { try studio.draft.validate(); throw NativeFailure(message: "mismatched source frame count was accepted") }
        catch let error as NativeFailure { try check(error.message.contains("完整源帧数"), "wrong frame-count rejection") }
        studio.draft.frames = 25
        studio.save()
        let restored = StudioState(directory: root)
        try check(restored.draft.activeAssets.first?.mediaKind == "video" && restored.draft.frames == 25,
                  "video source metadata/draft persistence failed")
        let legacy = StudioAsset(path: "legacy.png", name: "legacy", width: 64, height: 64)
        let object = try JSONDecoder().decode(StudioAsset.self, from: JSONEncoder().encode(legacy))
        try check(object.mediaKind == "image", "legacy image assets no longer decode")
        print("PASS SeedVR2 App image/video request kinds, embedded conditioning, video staging/FPS/frames, mismatch rejection and legacy asset persistence")
        if CommandLine.arguments.count >= 5 {
            let modelURL = URL(fileURLWithPath: CommandLine.arguments[3])
            let realImage = URL(fileURLWithPath: CommandLine.arguments[4])
            let jobs = NativeJobStore(directory: root.appendingPathComponent("jobs"))
            var results: [[String: Any]] = []
            for operation in ["image.upscale", "video.upscale"] {
                studio.changeOperation(operation)
                studio.draft.modelPaths[model.id] = modelURL.path
                await studio.addFiles([operation == "image.upscale" ? realImage : URL(fileURLWithPath: CommandLine.arguments[1])])
                try check(studio.message == nil, "real-media import failed: \(studio.message ?? "")")
                studio.draft.prompt = "Ignored old draft prompt"
                studio.draft.acceleration = StudioAcceleration(policy: execution, manifest: manifest ?? "")
                studio.draft.residency = residency
                studio.draft.seedvr2VAE = vaeOptions
                studio.draft.seedText = "42"
                for size in [512, 1024] {
                    studio.draft.width = size; studio.draft.height = size
                    let isVideo = operation == "video.upscale"
                    let output = root.appendingPathComponent("app-\(isVideo ? "video" : "image")-\(size).\(isVideo ? "mp4" : "png")")
                    let request = try studio.draft.request(output: output)
                    let job = try await jobs.generate(modelURL: modelURL, request: request)
                    try check(job.state == "succeeded" && !jobs.busy, "App generation did not succeed")
                    guard let json = job.resultJSON, let data = json.data(using: .utf8),
                          let metrics = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                        throw NativeFailure(message: "App generation metrics missing")
                    }
                    let expectedDecoder = isVideo ? (vaeOptions?.decoder ?? "buffered") : "buffered"
                    let expectedEncoder = isVideo ? (vaeOptions?.encoder ?? "buffered") : "buffered"
                    let expectedSpatial = vaeOptions?.spatial ?? "full"
                    let expectedEncoderSpatial = vaeOptions?.encoder_spatial ?? "full"
                    let fileRows = expectedSpatial == "file_rows"
                    let encoderFileRows = expectedEncoderSpatial == "file_rows"
                    let streamingInput = expectedEncoder == "streaming"
                    try check(metrics["vae_decoder_recipe"] as? String == expectedDecoder &&
                              metrics["vae_encoder_recipe"] as? String == expectedEncoder &&
                              metrics["vae_spatial_recipe"] as? String == expectedSpatial &&
                              metrics["vae_encoder_spatial_recipe"] as? String == expectedEncoderSpatial &&
                              metrics["streamed_input_frames"] as? Int == (streamingInput ? 25 : 0) &&
                              metrics["streamed_output_frames"] as? Int == (expectedDecoder == "streaming" ? 25 : 0) &&
                              metrics["vae_conv_recipe"] as? String == (vaeOptions?.convolution ?? "singleton2d"),
                              "App VAE execution differs from its selected configuration")
                    let expectedExecution = (causalExperiment || upscaleCausalExperiment) && isVideo ? "gpu_vae_cpu_ne_experimental" : execution
                    try check(metrics["operation"] as? String == operation && metrics["width"] as? Int == size &&
                              metrics["height"] as? Int == size && metrics["execution"] as? String == expectedExecution &&
                              metrics["checkpoint_sha256_verified"] as? Bool == true,
                              "App result identity/geometry mismatch")
                    try check(metrics["vae_history_recipe"] as? String == (vaeOptions?.history ?? "compact") &&
                              metrics["vae_pointwise_recipe"] as? String == "matmul",
                              "App did not execute the qualified VAE history/pointwise defaults")
                    let sharesUpscale = vaeOptions?.share_upscale_history ?? false
                    let sharedFrames = sharesUpscale && isVideo ? (metrics["padded_frames"] as? Int ?? 0) : 0
                    try check(metrics["vae_share_upscale_history"] as? Bool == sharesUpscale &&
                              metrics["vae_shared_upscale_frames"] as? Int == sharedFrames &&
                              (metrics["vae_shared_upscale_logical_bytes"] as? NSNumber)?.uint64Value == UInt64(sharedFrames*size*size*256*2),
                              "App upscale-sharing execution/byte scope differs from request")
                    let expectedRecipe = (hybrid ? "seedvr2-original-fp16-coreml-prefix-v1" : "seedvr2-stable-fp16-fp32-v1") +
                        (encoderFileRows ? "+vae-encoder-head-globalstats-file-rows-v1" : "") +
                        (fileRows ? "+vae-globalstats-file-rows-v1" : "") +
                        (causalExperiment && isVideo ? "+experimental-vae-causal-coreml-original-fp16-v1" : "") +
                        (upscaleCausalExperiment && isVideo ? "+experimental-vae-upscale-causal-coreml-original-fp16-v1" : "")
                    try check(metrics["numerical_recipe"] as? String == expectedRecipe &&
                              metrics["dit_projection_dtype"] as? String == "fp16" &&
                              metrics["dit_residual_dtype"] as? String == "fp32" &&
                              metrics["dit_velocity_dtype"] as? String == "fp32" &&
                              metrics["dit_attention_dtype"] as? String == "fp16",
                              "App did not execute the qualified stable mixed-precision profile")
                    if fileRows {
                        try check(metrics["vae_spatial_rows"] as? Int == (vaeOptions?.spatial_rows ?? 64) &&
                                  (metrics["vae_spatial_disk_budget_bytes"] as? NSNumber)?.uint64Value == (vaeOptions?.spatial_disk_budget_bytes ?? (8 << 30)) &&
                                  metrics["vae_spatial_file_live_bytes_after_decode"] as? Int == 0 &&
                                  metrics["vae_spatial_host_live_bytes_after_decode"] as? Int == 0 &&
                                  metrics["vae_spatial_cleanup_failures"] as? Int == 0,
                                  "App spatial schedule/retirement differs from request")
                        if size == 1024 {
                            try check((metrics["vae_spatial_tile_convolutions"] as? Int ?? 0) > 0 &&
                                      (metrics["vae_spatial_global_normalizations"] as? Int ?? 0) > 0,
                                      "1024 App spatial route never executed its eligible stages")
                            let written = (metrics["vae_spatial_file_written_bytes"] as? NSNumber)?.uint64Value ?? 0
                            try check(isVideo ? written > 0 : written == 0,"App spatial history storage differs from image/video lifetime")
                        }
                    }
                    if encoderFileRows {
                        try check(metrics["vae_encoder_spatial_file_live_bytes_after_encode"] as? Int == 0 &&
                                  metrics["vae_encoder_spatial_host_live_bytes_after_encode"] as? Int == 0 &&
                                  metrics["vae_encoder_spatial_cleanup_failures"] as? Int == 0,
                                  "App encoder file/window ownership did not retire")
                        if size == 1024 {
                            try check((metrics["vae_encoder_spatial_tile_convolutions"] as? Int ?? 0) > 0 &&
                                metrics["vae_encoder_spatial_global_normalizations"] as? Int == (isVideo ? 100 : 4),
                                "App encoder head did not execute complete global statistics")
                            let written = (metrics["vae_encoder_spatial_file_written_bytes"] as? NSNumber)?.uint64Value ?? 0
                            try check(isVideo ? written > 0 : written == 0,"App encoder causal files differ from image/video lifetime")
                        }
                    }
                    let precision = hybrid
                        ? "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_coreml_ffn_prefix+fp32_ffn_join+fp16_vae+fp32_euler"
                        : "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_vae+fp32_euler"
                    let plan = metrics["plan"] as? [String: Any]
                    try check(metrics["runtime_precision"] as? String == precision &&
                              plan?["precision"] as? String == precision,
                              "App plan and executed precision disagree")
                    let staged = residency == "component_staged"
                    let inputRecipe = isVideo ? (streamingInput ? "bounded_rgb_reader_and_fp16_encoder_chunks" : "normalized_rgb_only") :
                        (staged ? "normalized_rgb_plus_owned_alpha" : "normalized_rgb_plus_source_rgba")
                    try check(metrics["media_input_retention_recipe"] as? String == inputRecipe &&
                              (metrics["media_retained_alpha_bytes"] as? NSNumber)?.uint64Value == (isVideo ? 0 : UInt64(size*size*4)) &&
                              (metrics["media_retained_source_rgba_bytes"] as? NSNumber)?.uint64Value == (isVideo || staged ? 0 : UInt64(size*size*16)),
                              "App input buffer policy or alpha retention differs from the request")
                    let ownedInput = isVideo || staged
                    let releaseRecipe = streamingInput ? "bounded_chunks_and_source_eof_before_sampling" : (ownedInput ? "after_vae_encode_owned" : "borrowed")
                    try check(metrics["normalized_input_release_recipe"] as? String == releaseRecipe &&
                              (metrics["normalized_input_released_logical_bytes"] as? NSNumber)?.uint64Value == (!streamingInput && ownedInput ? UInt64(size*size*request.frames*3*4) : 0),
                              "App did not release its declared owned normalized input")
                    if streamingInput {
                        try check((metrics["source_rgb24_buffer_capacity_peak_bytes"] as? NSNumber)?.uint64Value == UInt64(512*512*3) &&
                                  metrics["vae_encoder_input_schedule"] as? String == "initial5_then4_with_last_frame_padding" &&
                                  metrics["vae_encoder_wall_includes_frame_import"] as? Bool == true,
                                  "App did not execute the bounded source/padding schedule")
                    }
                    let timings = metrics["timings_seconds"] as? [String: Any]
                    try check(plan?["residency"] as? String == residency &&
                              metrics["weight_residency_recipe"] as? String == (staged ? "stage_dit" : "resident") &&
                              metrics["dit_reloaded_for_sampling"] as? Bool == staged,
                              "App planned/executed weight residency mismatch")
                    let components = (staged ? ["vae"] : ["vae", "dit"]) + (hybrid ? ["coreml_ffn_prefix"] : []) +
                        (causalExperiment ? ["experimental_coreml_vae_causal"] : []) +
                        (upscaleCausalExperiment ? ["experimental_coreml_vae_upscale_causal"] : [])
                    if causalExperiment {
                        let expectedCalls = isVideo ? 24 * ((size + 63) / 64) * ((size + 63) / 64) : 0
                        try check(metrics["vae_causal_conv_experimental_enabled"] as? Bool == true &&
                            metrics["vae_causal_conv_calls"] as? Int == expectedCalls &&
                            metrics["vae_causal_conv_backing_hits"] as? Int == expectedCalls &&
                            metrics["vae_causal_conv_frames"] as? Int == (isVideo ? 24 : 0) &&
                            metrics["vae_causal_conv_manifest_sha256"] as? String == "f9c751ba3f0d225ae9aaf2f926e1f63ed93ac44583de5d1ab57b79c290d26a08" &&
                            metrics["physical_ane_occupancy_measured"] as? Bool == false,
                            "experimental App VAE causal execution differs from actual coverage")
                    } else {
                        try check(metrics["vae_causal_conv_experimental_enabled"] as? Bool != true,"ordinary App executed an experimental VAE route")
                    }
                    if upscaleCausalExperiment {
                        let expectedCalls = isVideo ? 24 * ((size + 63) / 64) * ((size + 63) / 64) : 0
                        try check(metrics["vae_upscale_causal_conv_experimental_enabled"] as? Bool == true &&
                            metrics["vae_upscale_causal_conv_layer"] as? String == "decoder.up_blocks.2.upsamplers.0.conv" &&
                            metrics["vae_upscale_causal_conv_output_channels"] as? Int == 256 &&
                            metrics["vae_upscale_causal_conv_calls"] as? Int == expectedCalls &&
                            metrics["vae_upscale_causal_conv_backing_hits"] as? Int == expectedCalls &&
                            metrics["vae_upscale_causal_conv_frames"] as? Int == (isVideo ? 24 : 0) &&
                            metrics["vae_upscale_causal_conv_manifest_sha256"] as? String == "877f50e7d97399b3eae2e2c90b6832b8d99bb9d547e699683d37631d2060598b" &&
                            metrics["physical_ane_occupancy_measured"] as? Bool == false,
                            "experimental App upscale causal execution differs from full layer coverage")
                    } else {
                        try check(metrics["vae_upscale_causal_conv_experimental_enabled"] as? Bool != true,"App unexpectedly executed upscale causal offload")
                    }
                    try check(metrics["resident_components"] as? [String] == components,
                              "App reported the wrong resident components")
                    let management = (timings?["weight_management"] as? NSNumber)?.doubleValue ?? -1
                    try check(staged ? management > 0 : management == 0, "App omitted or misreported staging cost")
                    if staged {
                        try check(metrics["dit_weights_reused"] as? Bool == false &&
                                  (metrics["resident_parameter_bytes"] as? NSNumber)?.uint64Value == 501_296_454,
                                  "staged request retained DiT parameters or claimed their reuse")
                    }
                    if hybrid {
                        let tokens = ((request.frames - 1) / 4 + 1) * (size / 16) * (size / 16)
                        let calls = 32 * ((tokens + 511) / 512)
                        try check(metrics["coreml_prefix_calls"] as? Int == calls &&
                                  metrics["coreml_prefix_video_blocks"] as? Int == 32 &&
                                  metrics["coreml_prefix_compute_units"] as? String == "cpuAndNeuralEngine" &&
                                  metrics["coreml_prefix_manifest_sha256"] as? String == "109dde47f82b7acb3a6d8cbba16b9d42e9f15fddae539d64141a4ec0ac8a8f76" &&
                                  metrics["physical_ane_occupancy_measured"] as? Bool == false,
                                  "App hybrid did not execute its declared partition or overstated occupancy")
                        let approximations = plan?["algorithm_approximations"] as? [String] ?? []
                        try check(approximations.contains("seedvr2_original_fp16_video_ffn_prefix_fp16_partial_down_fp32_join") &&
                                  !approximations.contains("single_block_mlp_coreml_approximation"),
                                  "App mislabeled its all-block SeedVR2 approximation")
                    } else {
                        try check(metrics["coreml_prefix_calls"] as? Int == 0, "pure GPU request executed a Core ML prefix")
                    }
                    if isVideo {
                        let inspected = try await StudioAssetImporter(directory: root.appendingPathComponent("verified-media")).importVideoFile(output)
                        try check(inspected.width == size && inspected.height == size && inspected.frames == 25 && inspected.fps == 24,
                                  "App output MP4 frames/FPS/dimensions mismatch")
                    } else {
                        guard let image = NSBitmapImageRep(data: try Data(contentsOf: output)) else { throw NativeFailure(message: "App output PNG invalid") }
                        try check(image.pixelsWide == size && image.pixelsHigh == size, "App output PNG dimensions mismatch")
                    }
                    results.append(metrics)
                    studio.save()
                    let reloaded = NativeJobStore(directory: root.appendingPathComponent("jobs"))
                    try check(reloaded.jobs.first?.id == job.id && reloaded.jobs.first?.state == "succeeded",
                              "real App job history did not persist")
                    print("PASS real SeedVR2 App \(operation) \(size): \(json)")
                }
            }
            // Exercise transitions within the same real job-store/session
            // workflow. A resident request must reload the released bank;
            // a staged request must not inherit an old resident policy.
            studio.changeOperation("image.upscale")
            await studio.addFiles([realImage])
            studio.draft.width = 512; studio.draft.height = 512
            var referencePNG: Data?
            for (index, policy) in ["resident", "component_staged", "resident"].enumerated() {
                studio.draft.residency = policy
                let output = root.appendingPathComponent("policy-switch-\(index).png")
                let request = try studio.draft.request(output: output)
                let job = try await jobs.generate(modelURL: modelURL, request: request)
                guard let json = job.resultJSON, let bytes = json.data(using: .utf8),
                      let metrics = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
                    throw NativeFailure(message: "policy-switch result missing")
                }
                try check(job.state == "succeeded" &&
                          metrics["weight_residency_recipe"] as? String == (policy == "resident" ? "resident" : "stage_dit") &&
                          metrics["execution"] as? String == execution,
                          "policy switch did not execute the requested residency")
                let image = try Data(contentsOf: output)
                if let referencePNG { try check(referencePNG == image, "residency switch changed PNG bytes") }
                else { referencePNG = image }
                results.append(metrics)
            }
            try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
                .write(to: root.appendingPathComponent("generation-report.json"), options: .atomic)
            try await jobs.unload()
        }
    }
}

private extension Optional {
    func unwrap(_ message: String) throws -> Wrapped {
        guard let value = self else { throw NativeFailure(message: message) }
        return value
    }
}
