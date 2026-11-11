import AppKit
import Combine
import Foundation

@main struct KeyframeWorkflowTests {
    @MainActor static func main() async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try condition() else { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected invalid keyframe draft to be rejected")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-keyframes-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let oldLibrary = ProcessInfo.processInfo.environment["TURBOCIDER_MODEL_LIBRARY"]
        setenv("TURBOCIDER_MODEL_LIBRARY", root.appendingPathComponent("library").path, 1)
        defer {
            if let oldLibrary { setenv("TURBOCIDER_MODEL_LIBRARY", oldLibrary, 1) }
            else { unsetenv("TURBOCIDER_MODEL_LIBRARY") }
        }
        func image(_ name: String, red: UInt8, blue: UInt8) throws -> URL {
            let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 64,
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
            for y in 0..<64 { for x in 0..<64 {
                let p = bitmap.bitmapData!.advanced(by: y * bitmap.bytesPerRow + x * 4)
                p[0] = red; p[1] = 30; p[2] = blue; p[3] = 255
            } }
            let url = root.appendingPathComponent(name)
            try bitmap.representation(using: .png, properties: [:])!.write(to: url)
            return url
        }
        let firstURL = try image("first.png", red: 200, blue: 40)
        let lastURL = try image("last.png", red: 40, blue: 200)
        let studio = StudioState(directory: root.appendingPathComponent("state"))
        studio.selectModel("ltx-2.5-distilled")
        try check(studio.draft.residency == "streamed", "New LTX workflows must default to block streaming")
        studio.draft.modelPaths["ltx-2.5-distilled"] = root.path
        studio.draft.prompt = "The camera moves smoothly between the supplied frames."
        studio.draft.streaming.selection = .tier12
        studio.draft.ltxBackend = "cpp_mlx"
        studio.draft.ltxAccelerationMode = "fast_approx"
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
        studio.changeOperation("video.keyframes")
        try check(studio.draft.modelID == "ltx-2.5-distilled" && studio.draft.operation == "video.keyframes",
                  "Selecting LTX keyframes silently selected another model")
        try check(studio.draft.ltxBackend == "c_metal" && studio.draft.ltxAccelerationMode == "quality" &&
                  studio.draft.streaming.selection == .off && !studio.draft.usesANE && studio.draft.strength == 1,
                  "Keyframe selection retained incompatible execution settings")
        let firstImported = await studio.importKeyframe(firstURL, role: .first)
        try check(firstImported && studio.draft.firstKeyframe != nil && studio.draft.lastKeyframe == nil,
                  "First slot did not bind independently")
        try rejects { try studio.draft.validate() }
        let lastImported = await studio.importKeyframe(lastURL, role: .last)
        try check(lastImported && studio.draft.activeAssets.count == 2, "Tail slot did not complete the endpoint pair")
        let first = studio.draft.firstKeyframe!, last = studio.draft.lastKeyframe!
        try check(first.path != firstURL.path && last.path != lastURL.path &&
                  (try Data(contentsOf: URL(fileURLWithPath: first.path))) == (try Data(contentsOf: firstURL)),
                  "Keyframe inputs were not independent private copies")
        func request() throws -> NativeRequest { try studio.draft.request(output: root.appendingPathComponent("clip.mp4")) }
        let pair = try request()
        try check(pair.residency == "streamed" && pair.memory_budget_bytes == UInt64(12) << 30 && LTXWorker.accepts(pair),
                  "Keyframes lost the bounded isolated streaming worker route")
        try check(pair.inputs?.map(\.role) == ["first_frame", "last_frame"] &&
                  pair.inputs?.map(\.path) == [first.path, last.path] && pair.inputs?.map(\.strength) == [1, 1],
                  "Endpoint request roles, order or shared strength are incorrect")
        studio.draft.assets.reverse()
        try check(try request().inputs?.map(\.path) == [first.path, last.path],
                  "Tray order silently changed endpoint roles")
        studio.swapKeyframes()
        try check(try request().inputs?.map(\.path) == [last.path, first.path], "Swap did not reverse endpoint direction")
        studio.undoAssetChange()
        try check(studio.draft.firstKeyframe?.id == first.id && studio.draft.lastKeyframe?.id == last.id,
                  "Undo failed to restore endpoint bindings")
        studio.removeKeyframe(.first)
        try check(studio.draft.firstKeyframe == nil && studio.draft.lastKeyframe?.id == last.id,
                  "Removing the first frame silently promoted the tail")
        try rejects { try studio.draft.validate() }
        studio.undoAssetChange()
        studio.removeKeyframe(.last)
        studio.draft.assets.append(StudioAsset(path: lastURL.path, name: "unused", width: 64, height: 64))
        studio.save()
        let reopened = StudioState(directory: root.appendingPathComponent("state"))
        try check(reopened.draft.firstKeyframe?.id == first.id && reopened.draft.lastKeyframe == nil,
                  "Reopening an explicitly empty tail slot assigned an unrelated image")
        studio.undoAssetChange()
        var legacy = try JSONSerialization.jsonObject(with: JSONEncoder().encode(studio.draft)) as! [String: Any]
        legacy.removeValue(forKey: "lastImageID"); legacy.removeValue(forKey: "keyframeBindingsVersion")
        let migrated = try JSONDecoder().decode(StudioDraft.self, from: JSONSerialization.data(withJSONObject: legacy))
        try check(migrated.firstKeyframe?.id == migrated.assets[0].id && migrated.lastKeyframe?.id == migrated.assets[1].id,
                  "Older ordered keyframe drafts were not migrated")
        for operation in ["video.generate", "video.image", "video.keyframes"] {
            for audio in [false, true] {
                var old = studio.draft
                old.operation = operation; old.audio = audio; old.residency = "resident"
                var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as! [String: Any]
                json.removeValue(forKey: "ltxStreamingDefaultVersion")
                let restored = try JSONDecoder().decode(StudioDraft.self, from: JSONSerialization.data(withJSONObject: json))
                let converted = try restored.request(output: root.appendingPathComponent("restored.mp4"))
                try check(restored.residency == "streamed" && converted.memory_budget_bytes == UInt64(12) << 30 &&
                          LTXWorker.accepts(converted) && converted.audio == audio && restored.prompt == old.prompt &&
                          restored.assets == old.assets, "Old GPU draft did not migrate safely: \(operation), audio=\(audio)")
                var optedOut = restored; optedOut.residency = "resident"
                let reopened = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(optedOut))
                try check(reopened.residency == "resident", "Explicit current-version loading choice was overwritten")
                var historicalRequest = converted
                historicalRequest.residency = "resident"; historicalRequest.memory_budget_bytes = nil
                let historical = NativeJob(id: UUID(), createdAt: Date(), request: historicalRequest, state: "succeeded",
                    phase: "complete", completed: 11, total: 11, elapsed: 1, modelPath: root.path,
                    inputAssets: restored.activeAssets)
                let reuse = StudioState(directory: root.appendingPathComponent("reuse-\(operation)-\(audio)"))
                reuse.reuse(historical)
                try check(reuse.draft.residency == "streamed" && reuse.draft.audio == audio && reuse.draft.prompt == old.prompt,
                          "Old history restored an unbounded resident route")
            }
        }
        studio.setLTXResidency("resident")
        let optedOut = StudioState(directory: root.appendingPathComponent("state"))
        try check(optedOut.draft.residency == "resident", "Explicit resident selection was not durable")
        studio.setLTXResidency("streamed")
        try check(studio.draft.firstKeyframe?.id == first.id && studio.draft.lastKeyframe?.id == last.id &&
                  studio.draft.accelerationHint.contains("流式加载已开启"), "Enabling streaming lost endpoints or visible loading status")
        studio.setANEEnabled(true)
        try check(!studio.draft.usesANE && studio.draft.residency == "streamed", "Keyframes admitted an incompatible ANE route")
        let job = NativeJob(id: UUID(), createdAt: Date(), request: try request(), state: "succeeded",
            phase: "complete", completed: 11, total: 11, elapsed: 1, modelPath: root.path,
            inputAssets: studio.draft.activeAssets)
        let reuse = StudioState(directory: root.appendingPathComponent("reuse"))
        reuse.reuse(job)
        try check(reuse.draft.firstKeyframe?.path == first.path && reuse.draft.lastKeyframe?.path == last.path,
                  "History reuse lost first/last roles or input snapshots")
        func snapshot() throws -> Data {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            return try encoder.encode(studio.draft)
        }
        func fileCount() throws -> Int { try FileManager.default.contentsOfDirectory(atPath: studio.importer.directory.path).count }
        let before = try snapshot(), count = try fileCount()
        let missing = await studio.importKeyframe(root.appendingPathComponent("missing.png"), role: .last)
        try check(!missing && (try snapshot()) == before && (try fileCount()) == count,
                  "A missing replacement changed endpoint bindings or leaked a copy")
        var completions = 0
        studio.beginKeyframeImport(firstURL, role: .last) { completions += 1 }
        let repeated = studio.beginKeyframeImport(lastURL, role: .last) { completions += 1 }
        studio.cancelImageImport()
        await studio.imageImportTask?.value
        try check(!repeated && completions == 0 && !studio.imageInputsBusy && (try snapshot()) == before &&
                  (try fileCount()) == count, "Queued cancellation or repeated clicks published a frame")
        let conflict = studio.$importing.dropFirst().sink { importing in
            if importing { studio.draft.lastImageID = nil }
        }
        let conflicted = await studio.importKeyframe(firstURL, role: .last)
        conflict.cancel()
        try check(!conflicted && studio.draft.lastImageID == nil && (try fileCount()) == count,
                  "An import overwrote changed role bindings or leaked its copy")
        try FileManager.default.removeItem(at: firstURL)
        try FileManager.default.removeItem(at: lastURL)
        try check(FileManager.default.fileExists(atPath: first.path) && FileManager.default.fileExists(atPath: last.path),
                  "Removing original images broke retained endpoint copies")
        print("PASS keyframe workflow: LTX capability, compatible settings, explicit roles, private copies, request order, swap/undo, empty-slot durability, legacy migration, history reuse, cancellation, conflicts and repeated clicks")
    }
}
