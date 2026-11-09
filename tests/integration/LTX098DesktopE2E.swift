import AppKit
import Foundation

/// Exercises the desktop App's real draft, media import, embedded job store,
/// video output and persisted history without requiring window automation.
@main struct LTX098DesktopE2E {
    @MainActor static func main() async throws {
        guard (4...5).contains(CommandLine.arguments.count),
              CommandLine.arguments.count == 4 || CommandLine.arguments[4] == "--resize-to-base" else {
            throw NativeFailure(message: "ltx098-desktop-e2e MODEL_ROOT FIRST_FRAME NEW_OUTPUT_DIRECTORY [--resize-to-base]")
        }
        let resizeToBase = CommandLine.arguments.count == 5
        let expectedSize = resizeToBase ? 512 : 640
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
        let modelID = "ltx-video-0.9.8-distilled"
        let studio = StudioState(directory: root)
        let store = NativeJobStore(directory: root)
        guard let entry = studio.models.first(where: { $0.id == modelID }) else {
            throw NativeFailure(message: "Desktop App model catalog lacks LTX 0.9.8")
        }
        try check(entry.executor && entry.supports("video.image"), "Desktop catalog did not advertise image-to-video")
        studio.selectInstallation(modelID: modelID, path: model.path)
        studio.draft.prompt = "A lighthouse stands on dark coastal rocks at sunset. Ocean waves roll and break gently around the rocks, the light in the lantern glows, and clouds drift slowly across the warm evening sky. The camera remains steady."
        studio.draft.randomSeed = false; studio.draft.seedText = "42"
        studio.draft.ltx098ResizeToBase = resizeToBase
        await studio.addFiles([image])
        try check(studio.draft.modelID == modelID && studio.draft.modelPath == model.path &&
                  studio.draft.operation == "video.image" && studio.draft.activeAssets.count == 1 &&
                  studio.draft.initImageID == studio.draft.activeAssets.first?.id,
                  "Desktop model selection or first-frame import failed: \(studio.message ?? "unknown error")")
        let movie = root.appendingPathComponent("outputs/ltx098-app-generated.mp4")
        // Follow StudioView.generate(): resolve the draft, construct the
        // legacy/public request pair and submit through the same job store.
        let resolved = try await store.resolveAcceleration(studio.draft)
        studio.rememberAcceleration(resolved)
        let pair = try resolved.publicStreamingRequest(output: movie)
        let options = try resolved.runtimeOptions(store: store.directory)
        let request = pair.legacy
        try check(request.model == modelID && request.operation == "video.image" && request.execution == "gpu" &&
                  request.residency == "resident" && request.width == 512 && request.height == 512 &&
                  request.frames == 121 && request.fps == 24 && request.steps == 10 && request.audio == false &&
                  request.seed == 42 && request.inputs?.count == 1 && request.inputs?.first?.role == "first_frame" &&
                  pair.v2 == nil && options == nil,
                  "Desktop request did not preserve LTX image-to-video defaults")
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
                  result["execution"] as? String == "gpu" && result["actual_denoise_steps"] as? Int == 10 &&
                  result["workflow"] as? String == "multiscale-7+3" &&
                  result["ltx098_adain_after_upscale"] as? Bool == true &&
                  result["ltx098_resize_to_base"] as? Bool == resizeToBase &&
                  result["width"] as? Int == expectedSize && result["height"] as? Int == expectedSize &&
                  result["model"] as? String == modelID && result["operation"] as? String == "video.image",
                  "Desktop embedded video generation did not succeed")
        let plan = result["plan"] as? [String: Any]
        try check(plan?["decoded_shape"] as? [Int] == [640, 640, 121] &&
                  plan?["output_shape"] as? [Int] == [expectedSize, expectedSize, 121],
                  "Desktop plan did not describe the actual multiscale geometry")
        let video = try await StudioAssetImporter(directory: root.appendingPathComponent("validated-media")).importVideoFile(movie)
        try check(video.width == expectedSize && video.height == expectedSize && video.frames == 121 && video.fps == 24 &&
                  job.outputDimensions.width == expectedSize && job.outputDimensions.height == expectedSize,
                  "Desktop-generated MP4 has unexpected geometry or duration")
        let reloaded = NativeJobStore(directory: root)
        try check(reloaded.jobs.first?.id == job.id && reloaded.jobs.first?.state == "succeeded",
                  "Desktop video job was not persisted")
        try await store.unload()
        let report: [String: Any] = ["schema": "ltx098-desktop-e2e-v1", "status": "complete",
            "scope": "Desktop StudioState, image importer and embedded NativeJobStore; no GUI window automation",
            "output": movie.path, "seed": request.seed, "first_frame_strength": request.inputs?.first?.strength ?? -1,
            "width": video.width, "height": video.height, "frames": video.frames ?? 0,
            "fps": video.fps ?? 0, "wall_seconds": elapsed, "result": result]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print(String(decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), as: UTF8.self))
    }
}
