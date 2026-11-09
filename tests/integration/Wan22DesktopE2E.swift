import AppKit
import Foundation

/// Exercises the desktop draft, image import, native job store, 97-frame MP4
/// output and persisted history; does not claim GUI-window automation.
@main struct Wan22DesktopE2E {
    @MainActor static func main() async throws {
        guard (4...5).contains(CommandLine.arguments.count),
              CommandLine.arguments.count == 4 || ["full", "super"].contains(CommandLine.arguments[4]) else {
            throw NativeFailure(message: "wan22-desktop-e2e MODEL_ROOT FIRST_FRAME NEW_OUTPUT_DIRECTORY [full|super]")
        }
        let decoder = CommandLine.arguments.count == 5 ? CommandLine.arguments[4] : "default"
        let useSuper = decoder != "full"
        let model = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
        let image = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
        let root = URL(fileURLWithPath: CommandLine.arguments[3]).standardizedFileURL
        let files = FileManager.default
        guard !files.fileExists(atPath: root.path), files.fileExists(atPath: image.path) else {
            throw NativeFailure(message: "A new output directory and existing first-frame image are required")
        }
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw NativeFailure(message: message) }
        }
        let variant = ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_VARIANT"] ?? "fastwan"
        guard ["fastwan", "quanhaol"].contains(variant) else { throw NativeFailure(message: "Invalid Wan2.2 E2E variant") }
        let modelID = "wan2.2-ti2v-5b-\(variant)"
        let expectedSteps = variant == "quanhaol" ? 4 : 3
        let studio = StudioState(directory: root)
        let store = NativeJobStore(directory: root)
        guard let entry = studio.models.first(where: { $0.id == modelID }) else {
            throw NativeFailure(message: "Desktop catalog lacks Wan2.2")
        }
        try check(entry.executor && entry.supports("video.image"), "Desktop catalog did not advertise image-to-video")
        if ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_REQUIRE_SIBLING_REUSE"] == "1" {
            try check(variant == "quanhaol", "Sibling-reuse smoke test requires the quanhaol variant")
            studio.selectInstallation(modelID: "wan2.2-ti2v-5b-fastwan", path: model.path)
            studio.selectModel(modelID)
        } else {
            studio.selectInstallation(modelID: modelID, path: model.path)
        }
        studio.draft.prompt = "A red fox walks slowly toward the camera through a snowy field. Its fur moves gently in the wind and its paws step naturally in the snow. Keep the camera steady and the fox's face and anatomy consistent."
        studio.draft.randomSeed = false
        studio.draft.seedText = "42"
        if let snapshot = ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_E2E_JOB"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: snapshot))
            guard let jobs = try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
                  let saved = jobs.first?["request"] as? [String: Any],
                  let prompt = saved["prompt"] as? String, let seed = saved["seed"] as? NSNumber else {
                throw NativeFailure(message: "Invalid Wan2.2 E2E job snapshot")
            }
            studio.draft.prompt = prompt
            studio.draft.seedText = seed.stringValue
        }
        let width = Int(ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_E2E_WIDTH"] ?? "512") ?? 0
        let height = Int(ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_E2E_HEIGHT"] ?? "512") ?? 0
        guard let expectedSeed = Int(studio.draft.seedText) else { throw NativeFailure(message: "Invalid E2E seed") }
        studio.draft.width = width
        studio.draft.height = height
        if decoder == "full" { studio.draft.wan22TinyVAEDecoder = "original" }
        if decoder == "super" { studio.draft.wan22TinyVAEDecoder = "tiny" }
        await studio.addFiles([image])
        try check(studio.draft.modelID == modelID && studio.draft.modelPath == model.path &&
                  studio.draft.operation == "video.image" && studio.draft.activeAssets.count == 1 &&
                  studio.draft.initImageID == studio.draft.activeAssets.first?.id,
                  "Desktop model selection or first-frame import failed: \(studio.message ?? "unknown error")")
        let movie = root.appendingPathComponent("outputs/wan22-app-generated.mp4")
        let resolved = try await store.resolveAcceleration(studio.draft)
        studio.rememberAcceleration(resolved)
        let pair = try resolved.publicStreamingRequest(output: movie)
        let options = try resolved.runtimeOptions(store: store.directory)
        let request = pair.legacy
        try check(request.model == modelID && request.operation == "video.image" && request.execution == "gpu" &&
                  request.residency == "resident" && request.width == width && request.height == height &&
                  request.frames == 97 && request.fps == 24 && request.steps == expectedSteps && request.audio == false &&
                  request.seed == expectedSeed && request.inputs?.count == 1 && request.inputs?.first?.role == "first_frame" &&
                  request.inputs?.first?.strength == 1 && pair.v2 == nil && options == nil &&
                  (useSuper ? (request.tiny_vae?.encoder == "original" &&
                   request.tiny_vae?.decoder == "tiny" && request.allow_approximation == true) : request.tiny_vae == nil),
                  "Desktop request did not preserve Wan2.2 image-to-video defaults")
        if ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_REQUIRE_BUNDLED"] == "1" {
            guard let bundled = Bundle.main.resourceURL?.appendingPathComponent("tiny-vae/taew2_2_super.safetensors") else {
                throw NativeFailure(message: "App bundle resources are unavailable")
            }
            try check(useSuper && request.tiny_vae?.path == bundled.path && files.fileExists(atPath: bundled.path),
                      "Default Wan2.2 request did not select the packaged Super weights")
        }
        let started = ContinuousClock.now
        let job = try await store.generate(modelURL: model, request: request, streamingRequest: pair.v2,
                                           runtimeOptions: options, inputAssets: resolved.activeAssets)
        let duration = started.duration(to: .now).components
        let elapsed = Double(duration.seconds) + Double(duration.attoseconds) / 1e18
        guard let json = job.resultJSON, let data = json.data(using: .utf8),
              let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NativeFailure(message: "Desktop job did not provide a result receipt")
        }
        try check(job.state == "succeeded" && !store.busy && files.fileExists(atPath: movie.path) &&
                  result["execution"] as? String == "gpu" && result["actual_denoise_steps"] as? Int == expectedSteps &&
                  result["scheduler"] as? String == (variant == "quanhaol" ? "quanhaol-dmd-shift8-bf16" : "fastvideo-mlx-dmd-shift5-warped") &&
                  result["cross_attention_cache"] as? Bool == (ProcessInfo.processInfo.physicalMemory >= (64 * 1024 * 1024 * 1024)) &&
                  result["video_decoder"] as? String == (useSuper ? "taew2_2_super" : "wan2.2_vae") &&
                  result["model"] as? String == modelID && result["operation"] as? String == "video.image",
                  "Desktop embedded Wan2.2 generation did not succeed")
        let video = try await StudioAssetImporter(directory: root.appendingPathComponent("validated-media")).importVideoFile(movie)
        try check(video.width == width && video.height == height && video.frames == 97 && video.fps == 24,
                  "Desktop-generated MP4 has unexpected geometry or duration")
        if let expected = ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_EXPECT_TEXT_TOKENS"].flatMap(Int.init) {
            try check(result["valid_text_tokens"] as? Int == expected,
                      "Desktop job did not preserve the extended prompt")
        }
        if let expected = ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_EXPECT_TEXT_ALIGNMENT"].flatMap(Int.init) {
            try check(result["text_padding_alignment"] as? Int == expected,
                      "Desktop job used an unexpected text alignment")
        }
        if let expected = ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_EXPECT_ENCODER_TOKENS"].flatMap(Int.init) {
            try check(result["text_encoder_sequence_tokens"] as? Int == expected,
                      "Desktop job used an unexpected text encoder length")
        }
        let reloaded = NativeJobStore(directory: root)
        try check(reloaded.jobs.first?.id == job.id && reloaded.jobs.first?.state == "succeeded",
                  "Desktop video job was not persisted")
        var repeatReport: [String: Any]?
        if ProcessInfo.processInfo.environment["TURBOCIDER_WAN22_REPEAT_PROMPT"] == "1" {
            let repeatMovie = root.appendingPathComponent("outputs/wan22-app-repeat.mp4")
            let repeatPair = try resolved.publicStreamingRequest(output: repeatMovie)
            let repeatStart = ContinuousClock.now
            let repeated = try await store.generate(modelURL: model, request: repeatPair.legacy,
                streamingRequest: repeatPair.v2, runtimeOptions: options, inputAssets: resolved.activeAssets)
            let repeatDuration = repeatStart.duration(to: .now).components
            guard let repeatJSON = repeated.resultJSON, let repeatData = repeatJSON.data(using: .utf8),
                  let repeatResult = try JSONSerialization.jsonObject(with: repeatData) as? [String: Any] else {
                throw NativeFailure(message: "Repeated desktop job did not provide a result receipt")
            }
            try check(repeated.state == "succeeded" && repeatResult["prompt_cache_hit"] as? Bool == true &&
                      repeatResult["text_encoder_sequence_tokens"] as? Int == 0 &&
                      repeatResult["text_padding_alignment"] as? Int == result["text_padding_alignment"] as? Int &&
                      files.fileExists(atPath: repeatMovie.path),
                      "Repeated desktop job did not reuse its text conditioning")
            let repeatedVideo = try await StudioAssetImporter(directory: root.appendingPathComponent("validated-media")).importVideoFile(repeatMovie)
            try check(repeatedVideo.width == width && repeatedVideo.height == height &&
                      repeatedVideo.frames == 97 && repeatedVideo.fps == 24,
                      "Repeated desktop MP4 has unexpected geometry or duration")
            repeatReport = ["output": repeatMovie.path,
                "wall_seconds": Double(repeatDuration.seconds) + Double(repeatDuration.attoseconds) / 1e18,
                "result": repeatResult]
        }
        try await store.unload()
        var report: [String: Any] = ["schema": "wan22-desktop-e2e-v1", "status": "complete",
            "scope": "Desktop StudioState, image importer and embedded NativeJobStore; no GUI window automation",
            "output": movie.path, "seed": request.seed, "width": video.width, "height": video.height,
            "frames": video.frames ?? 0, "fps": video.fps ?? 0, "decoder": decoder, "variant": variant,
            "wall_seconds": elapsed, "result": result]
        if let repeatReport { report["repeated_job"] = repeatReport }
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print(String(decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), as: UTF8.self))
    }
}
