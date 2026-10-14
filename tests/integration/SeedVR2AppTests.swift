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
        let config = NativeSeedVR2VAEConfig(decoder: "streaming", convolution: "mixed3d", tileRows: 4096, allocatorCacheBytes: 1 << 30)
        studio.draft.seedvr2VAE = config
        let configured = try studio.draft.request(output: root.appendingPathComponent("configured.mp4"))
        try check(configured.seedvr2_vae == config && NativeRequestV2(legacy: configured).execution.seedvr2_vae == config,
                  "App/SDK lost request-local VAE controls")
        let restoredOptions = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(studio.draft))
        try check(restoredOptions.seedvr2VAE == config, "App draft did not persist VAE controls")
        studio.draft.seedvr2VAE = nil
        try check(video.operation == "video.upscale" && video.frames == 25 && video.fps == 24,
                  "source frame count/FPS were not applied to the draft")
        try check(video.inputs?.first?.kind == "video" && video.inputs?.first?.role == "source", "video source role/kind mismatch")
        try check(NativeRequestV2(legacy: video).outputs[0].kind == "video", "schema-2 video output kind mismatch")
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
                    try check(metrics["vae_decoder_recipe"] as? String == expectedDecoder &&
                              metrics["streamed_output_frames"] as? Int == (expectedDecoder == "streaming" ? 25 : 0) &&
                              metrics["vae_conv_recipe"] as? String == (vaeOptions?.convolution ?? "singleton2d"),
                              "App VAE execution differs from its selected configuration")
                    try check(metrics["operation"] as? String == operation && metrics["width"] as? Int == size &&
                              metrics["height"] as? Int == size && metrics["execution"] as? String == execution &&
                              metrics["checkpoint_sha256_verified"] as? Bool == true,
                              "App result identity/geometry mismatch")
                    try check(metrics["vae_history_recipe"] as? String == (vaeOptions?.history ?? "compact") &&
                              metrics["vae_pointwise_recipe"] as? String == "matmul",
                              "App did not execute the qualified VAE history/pointwise defaults")
                    try check(metrics["numerical_recipe"] as? String == (hybrid ? "seedvr2-original-fp16-coreml-prefix-v1" : "seedvr2-stable-fp16-fp32-v1") &&
                              metrics["dit_projection_dtype"] as? String == "fp16" &&
                              metrics["dit_residual_dtype"] as? String == "fp32" &&
                              metrics["dit_velocity_dtype"] as? String == "fp32" &&
                              metrics["dit_attention_dtype"] as? String == "fp16",
                              "App did not execute the qualified stable mixed-precision profile")
                    let precision = hybrid
                        ? "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_coreml_ffn_prefix+fp32_ffn_join+fp16_vae+fp32_euler"
                        : "original_fp16_weights+fp16_dit_projections+fp32_dit_state+fp16_vae+fp32_euler"
                    let plan = metrics["plan"] as? [String: Any]
                    try check(metrics["runtime_precision"] as? String == precision &&
                              plan?["precision"] as? String == precision,
                              "App plan and executed precision disagree")
                    let staged = residency == "component_staged"
                    let inputRecipe = isVideo ? "normalized_rgb_only" : (staged ? "normalized_rgb_plus_owned_alpha" : "normalized_rgb_plus_source_rgba")
                    try check(metrics["media_input_retention_recipe"] as? String == inputRecipe &&
                              (metrics["media_retained_alpha_bytes"] as? NSNumber)?.uint64Value == (isVideo ? 0 : UInt64(size*size*4)) &&
                              (metrics["media_retained_source_rgba_bytes"] as? NSNumber)?.uint64Value == (isVideo || staged ? 0 : UInt64(size*size*16)),
                              "App input buffer policy or alpha retention differs from the request")
                    let ownedInput = isVideo || staged
                    try check(metrics["normalized_input_release_recipe"] as? String == (ownedInput ? "after_vae_encode_owned" : "borrowed") &&
                              (metrics["normalized_input_released_logical_bytes"] as? NSNumber)?.uint64Value == (ownedInput ? UInt64(size*size*request.frames*3*4) : 0),
                              "App did not release its declared owned normalized input")
                    let timings = metrics["timings_seconds"] as? [String: Any]
                    try check(plan?["residency"] as? String == residency &&
                              metrics["weight_residency_recipe"] as? String == (staged ? "stage_dit" : "resident") &&
                              metrics["dit_reloaded_for_sampling"] as? Bool == staged,
                              "App planned/executed weight residency mismatch")
                    let components = (staged ? ["vae"] : ["vae", "dit"]) + (hybrid ? ["coreml_ffn_prefix"] : [])
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
