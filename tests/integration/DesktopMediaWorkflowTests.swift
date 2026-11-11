import AppKit
import Combine
import Foundation

/// UI workflow contracts using tiny images; no model weights or inference.
@main struct DesktopMediaWorkflowTests {
    @MainActor static func main() async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try condition() else { throw NativeFailure(message: message) }
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-desktop-media-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let oldLibrary = ProcessInfo.processInfo.environment["TURBOCIDER_MODEL_LIBRARY"]
        setenv("TURBOCIDER_MODEL_LIBRARY", root.appendingPathComponent("library").path, 1)
        defer {
            if let oldLibrary { setenv("TURBOCIDER_MODEL_LIBRARY", oldLibrary, 1) }
            else { unsetenv("TURBOCIDER_MODEL_LIBRARY") }
        }
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 96,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        let png = bitmap.representation(using: .png, properties: [:])!
        let source = root.appendingPathComponent("result.PNG")
        try png.write(to: source)
        var request = NativeRequest(prompt: "An image prompt", output: source.path)
        request.model = "qwen-image-2.1"; request.operation = "image.generate"
        let image = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded",
            phase: "complete", completed: 1, total: 1, elapsed: 1)
        var videoRequest = request
        videoRequest.model = "ltx-2.5-distilled"; videoRequest.operation = "video.generate"
        videoRequest.output = root.appendingPathComponent("clip.MP4").path
        let video = NativeJob(id: UUID(), createdAt: Date(), request: videoRequest, state: "succeeded",
            phase: "complete", completed: 1, total: 1, elapsed: 1)
        var deleted = NativeJob(id: UUID(), createdAt: Date(), request: videoRequest, state: "succeeded",
            phase: "complete", completed: 1, total: 1, elapsed: 1)
        deleted.outputDeleted = true
        let failed = NativeJob(id: UUID(), createdAt: Date(), request: videoRequest, state: "failed",
            phase: "complete", completed: 1, total: 1, elapsed: 1)
        let mixed = [video, image, deleted, failed]
        let videoHistory = mixed.filter { $0.hasOutput && $0.outputMediaKind == .video }
        let imageHistory = mixed.filter { $0.hasOutput && $0.outputMediaKind == .image }
        try check(videoHistory.map(\.id) == [video.id] && imageHistory.map(\.id) == [image.id],
                  "Mixed history did not separate media or included deleted/failed results")
        for ext in ["mp4", "MOV", "m4v", "webm"] {
            try check(StudioMediaKind(path: "/tmp/clip.\(ext)") == .video, "Video classification failed: \(ext)")
        }
        let historyRoot = root.appendingPathComponent("history")
        try FileManager.default.createDirectory(at: historyRoot, withIntermediateDirectories: true)
        try JSONEncoder().encode(mixed).write(to: historyRoot.appendingPathComponent("jobs.json"))
        let reopenedHistory = NativeJobStore(directory: historyRoot)
        try check(reopenedHistory.jobs.filter { $0.hasOutput && $0.outputMediaKind == .video }.map(\.id) == [video.id],
                  "Video history did not survive reopening")
        let studio = StudioState(directory: root.appendingPathComponent("studio"))
        studio.selectModel("qwen-image-2.1")
        studio.draft.modelPaths["ltx-2.5-distilled"] = root.appendingPathComponent("video-model").path
        studio.draft.prompt = request.prompt
        await studio.addFiles([source, source])
        let originalAssets = studio.draft.assets
        let imported = await studio.useResultForVideo(image)
        try check(imported && studio.draft.modelID == "ltx-2.5-distilled" && studio.draft.operation == "video.image",
                  "Result handoff did not select the installed image-to-video executor")
        try check(studio.draft.assets.count == 1 && studio.draft.activeAssets.count == 1 && studio.draft.prompt.isEmpty,
                  "Image handoff retained unrelated references or the image prompt")
        let copied = URL(fileURLWithPath: studio.draft.activeAssets[0].path)
        try check(copied != source && (try Data(contentsOf: copied)) == png && (try Data(contentsOf: source)) == png,
                  "Handoff modified the output or failed to stage a private copy")
        studio.draft.prompt = "Slow camera pan"
        let native = try studio.draft.request(output: root.appendingPathComponent("new-video.mp4"))
        try check(native.operation == "video.image" && native.inputs?.count == 1 &&
                  native.inputs?.first?.path == copied.path && native.inputs?.first?.role == "first_frame",
                  "Image-to-video request did not consume the copied first frame")
        let restored = StudioState(directory: root.appendingPathComponent("studio"))
        try check(restored.draft.operation == "video.image" && restored.draft.activeAssets.first?.path == copied.path,
                  "First-frame handoff was not durable")
        studio.undoAssetChange()
        try check(studio.draft.assets == originalAssets, "Handoff could not restore previous input references")

        // Choosing another first frame within video preserves motion settings.
        studio.draft.prompt = "Keep this motion prompt"; studio.draft.frames = 33
        studio.draft.steps = 8; studio.draft.fps = 16
        let replaced = await studio.useResultForVideo(image)
        try check(replaced && studio.draft.prompt == "Keep this motion prompt" && studio.draft.frames == 33 &&
                  studio.draft.steps == 8 && studio.draft.fps == 16,
                  "Replacing the first frame reset the existing video configuration")
        studio.draft.steps = 11; studio.draft.fps = 24
        studio.draft.streaming.selection = .tier12; studio.draft.streaming.userSelected = true
        studio.draft.ltxBackend = "cpp_mlx"
        let adjusted = await studio.useResultForVideo(image)
        let compatible = try studio.draft.publicStreamingRequest(output: root.appendingPathComponent("compatible.mp4"))
        try check(adjusted && studio.draft.streaming.selection == .off && studio.draft.ltxBackend == "c_metal" &&
                  compatible.v2 == nil && compatible.legacy.inputs?.first?.role == "first_frame",
                  "Handoff retained text-only streaming/MLX settings that cannot run image-to-video")
        func snapshot() throws -> Data {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            return try encoder.encode(studio.draft)
        }
        func stagedCount() throws -> Int { try FileManager.default.contentsOfDirectory(atPath: studio.importer.directory.path).count }
        let before = try snapshot(), count = try stagedCount()
        let invalidVideo = await studio.useResultForVideo(video)
        var deletedImage = image; deletedImage.outputDeleted = true
        let invalidDeleted = await studio.useResultForVideo(deletedImage)
        try check(!invalidVideo && !invalidDeleted && (try snapshot()) == before,
                  "A video/deleted result was imported as an image")
        var missingRequest = request; missingRequest.output = root.appendingPathComponent("missing.png").path
        let missing = NativeJob(id: UUID(), createdAt: Date(), request: missingRequest, state: "succeeded",
            phase: "complete", completed: 1, total: 1, elapsed: 1)
        let missingResult = await studio.useResultForVideo(missing)
        try check(!missingResult && (try snapshot()) == before && (try stagedCount()) == count,
                  "Missing source changed the draft or leaked a staged file")
        let cancellation = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            return await studio.useResultForVideo(image)
        }
        let cancelled = await cancellation.value
        try check(!cancelled && (try snapshot()) == before && (try stagedCount()) == count,
                  "Cancelled handoff changed the draft or leaked a staged file")
        let conflict = studio.$importing.dropFirst().sink { importing in
            if importing { studio.draft.assets.reverse() }
        }
        studio.draft.assets = originalAssets
        let conflicted = await studio.useResultForVideo(image)
        conflict.cancel()
        try check(!conflicted && studio.draft.assets == Array(originalAssets.reversed()) && (try stagedCount()) == count,
                  "Handoff overwrote concurrent input changes or leaked its copy")

        let noVideo = StudioState(directory: root.appendingPathComponent("no-video"),
            models: studio.models.filter { !$0.isVideo })
        noVideo.draft.prompt = "Preserve this draft"
        let unavailable = await noVideo.useResultForVideo(image)
        try check(!unavailable && noVideo.draft.prompt == "Preserve this draft" && noVideo.draft.assets.isEmpty,
                  "Missing video executor damaged the image draft")

        // The public entry latches immediately, rejects a second handoff and
        // invokes completion only after successful publication.
        var completions = 0
        let started = studio.beginVideoFromResult(image) { completions += 1 }
        let duplicate = studio.beginVideoFromResult(image) { completions += 1 }
        try check(started && !duplicate && studio.imageInputsBusy, "Handoff did not latch against repeated clicks")
        await studio.imageImportTask?.value
        try check(completions == 1 && !studio.imageInputsBusy, "Handoff completion or import cleanup failed")
        let beforeQueuedCancellation = try snapshot(), queuedCount = try stagedCount()
        studio.beginVideoFromResult(image) { completions += 1 }
        studio.cancelImageImport()
        await studio.imageImportTask?.value
        try check(completions == 1 && !studio.imageInputsBusy && (try snapshot()) == beforeQueuedCancellation &&
                  (try stagedCount()) == queuedCount, "Cancelling the queued handoff published a frame or retained its lock")
        let retained = URL(fileURLWithPath: studio.draft.activeAssets[0].path)
        try FileManager.default.removeItem(at: source)
        try check((try Data(contentsOf: retained)) == png, "Deleting the original result broke the video input")
        print("PASS desktop media: mixed/reopened history, first-frame request, copy ownership, undo, persistence, video settings, missing source/executor, cancellation, conflicts and duplicate clicks")
    }
}
