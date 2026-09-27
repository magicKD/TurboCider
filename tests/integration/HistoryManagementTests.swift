import Foundation
import AppKit

@main
struct HistoryManagementTests {
    @MainActor static func main() async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try condition() else { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected validation failure")
        }

        // Metadata-only plan labels in older results guessed INT8. Only a
        // loaded session's declared variant can supply the MLP precision.
        for (variant, expected) in [("fp16", "MLP FP16"), ("int8_pc", "MLP INT8"),
                                    ("unknown", "MLP 精度未确认"), ("", "MLP 精度未确认")] {
            var hybridJob = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "precision fixture", output: "/tmp/metadata-only.png"),
                state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1)
            var payload: [String: Any] = ["plan": ["execution": "gpu_ane_experimental",
                "precision": "bf16_gpu+int8_mlp_fp16_io"]]
            if !variant.isEmpty { payload["hybrid"] = ["weight_variant": variant] }
            hybridJob.resultJSON = String(decoding: try JSONSerialization.data(withJSONObject: payload), as: UTF8.self)
            try check(hybridJob.routeSummary == "GPU + Core ML · \(expected) · ANE 驻留未知",
                      "History guessed or lost hybrid precision: \(variant)")
        }

        let order = (0..<6).map { _ in UUID() }
        var selection = HistorySelection()
        selection.select(order[1], orderedIDs: order)
        selection.select(order[4], orderedIDs: order, toggle: true)
        try check(selection.ids == [order[1], order[4]], "Control/Command did not add an individual item")
        selection.select(order[4], orderedIDs: order, toggle: true)
        try check(selection.ids == [order[1]], "Control/Command did not deselect an item")
        selection.select(order[1], orderedIDs: order)
        selection.select(order[4], orderedIDs: order, range: true)
        try check(selection.ids == Set(order[1...4]), "Forward Shift range failed")
        selection.select(order[2], orderedIDs: order, range: true)
        try check(selection.ids == Set(order[1...2]), "Shift did not shrink the range from its original anchor")
        selection.select(order[4], orderedIDs: order)
        selection.select(order[1], orderedIDs: order, range: true)
        try check(selection.ids == Set(order[1...4]), "Reverse Shift range failed")
        selection.select(order[0], orderedIDs: order, toggle: true)
        selection.select(order[2], orderedIDs: order, toggle: true, range: true)
        try check(selection.ids == Set(order[0...4]), "Control/Command+Shift discarded existing selections")
        selection.retain(Set(order.dropFirst()))
        try check(selection.anchor == nil && !selection.ids.contains(order[0]), "Removed anchor was retained")
        selection.select(order[5], orderedIDs: Array(order.dropFirst()), range: true)
        try check(selection.ids == [order[5]], "Shift without an anchor did not select a single item")
        selection.selectAll(order)
        selection.select(UUID(), orderedIDs: order)
        try check(selection.ids == Set(order), "An unknown item changed the selection")
        selection.clear()
        try check(selection.ids.isEmpty && selection.anchor == nil, "Clear left a stale selection anchor")

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-history-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let outputs = root.appendingPathComponent("outputs")
        let history = root.appendingPathComponent("jobs.json")
        try FileManager.default.createDirectory(at: outputs, withIntermediateDirectories: true)
        let jobs = try (0..<4).map { index -> NativeJob in
            let output = outputs.appendingPathComponent("\(UUID()).png")
            try Data([UInt8(index)]).write(to: output)
            return NativeJob(id: UUID(), createdAt: Date(timeIntervalSince1970: Double(100 - index)),
                             request: NativeRequest(prompt: "History fixture \(index)", output: output.path),
                             state: index == 3 ? "failed" : "succeeded", phase: "complete",
                             completed: 1, total: 1, elapsed: 1)
        }
        func save(_ rows: [NativeJob]) throws { try JSONEncoder().encode(rows).write(to: history) }
        func exists(_ job: NativeJob) -> Bool { FileManager.default.fileExists(atPath: job.request.output) }
        try save(jobs)
        let store = NativeJobStore(directory: root)
        try store.deleteJobs([jobs[0].id, jobs[2].id])
        try check(store.jobs.map(\.id) == [jobs[1].id, jobs[3].id], "Batch deletion removed unselected tasks")
        try check(jobs.allSatisfy(exists), "Deleting task records removed result files")
        try check(NativeJobStore(directory: root).jobs.map(\.id) == store.jobs.map(\.id), "Batch deletion did not persist")
        try store.deleteJobs([])
        try check(store.deletedJobs.count == 2, "Empty deletion erased undo")
        try rejects { try store.deleteJobs([jobs[1].id, UUID()]) }
        try check(store.jobs.count == 2 && store.deletedJobs.count == 2, "Invalid batch partially changed history")
        try store.undoDeleteJob()
        try check(store.jobs.map(\.id) == jobs.map(\.id), "Batch undo did not restore records in order")
        try store.clearHistory()
        try store.clearHistory()
        try check(store.jobs.isEmpty && store.deletedJobs.count == 4 && jobs.allSatisfy(exists), "Clear history lost files or undo")
        try store.undoDeleteJob()
        try check(NativeJobStore(directory: root).jobs.map(\.id) == jobs.map(\.id), "Clear-history undo did not persist")

        // A directory in place of jobs.json forces a real atomic-write failure.
        try store.deleteJob(jobs[3].id)
        try FileManager.default.removeItem(at: history)
        try FileManager.default.createDirectory(at: history, withIntermediateDirectories: false)
        try rejects { try store.deleteJobs([jobs[0].id, jobs[1].id]) }
        try check(store.jobs.count == 3 && store.deletedJobs.map(\.id) == [jobs[3].id], "Failed history save changed records or undo")
        try rejects { try store.undoDeleteJob() }
        try check(store.jobs.count == 3 && store.deletedJobs.count == 1, "Failed undo changed records")
        try rejects { try store.trashOutputs([jobs[0].id, jobs[1].id]) }
        try check(jobs.allSatisfy(exists) && store.jobs.allSatisfy { $0.outputDeleted == nil }, "Failed image save did not restore files and metadata")
        try FileManager.default.removeItem(at: history)
        try save(store.jobs)
        try store.undoDeleteJob()

        let external = root.appendingPathComponent("external.png")
        try Data([9]).write(to: external)
        let externalJob = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "External", output: external.path),
                                    state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1)
        try save(jobs + [externalJob])
        let mixedStore = NativeJobStore(directory: root)
        try rejects { try mixedStore.trashOutputs([jobs[0].id, externalJob.id]) }
        try rejects { try mixedStore.trashOutputs([jobs[0].id, jobs[3].id]) }
        try check(jobs.allSatisfy(exists) && exists(externalJob), "Invalid image batch moved files before validation")

        let symlink = outputs.appendingPathComponent("link.png")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: external)
        let linkedJob = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "Linked", output: symlink.path),
                                 state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1)
        let aliasJob = NativeJob(id: UUID(), createdAt: Date(), request: jobs[0].request,
                                state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1)
        try save(jobs + [linkedJob, aliasJob])
        let imageStore = NativeJobStore(directory: root)
        try rejects { try imageStore.trashOutputs([jobs[0].id, linkedJob.id]) }
        try check(jobs.allSatisfy(exists), "A symlink bypassed managed-output validation")
        let originalPaths = [jobs[0].request.output, jobs[1].request.output].sorted()
        let trashed = try imageStore.trashOutputs([jobs[0].id, jobs[1].id, aliasJob.id])
        defer {
            for (source, path) in zip(trashed, originalPaths) { try? FileManager.default.moveItem(at: source, to: URL(fileURLWithPath: path)) }
        }
        try check(trashed.count == 2 && !exists(jobs[0]) && !exists(jobs[1]) && exists(jobs[2]), "Image batch moved the wrong files")
        let persisted = NativeJobStore(directory: root)
        let deletedIDs: Set<UUID> = [jobs[0].id, jobs[1].id, aliasJob.id]
        try check(Set(persisted.jobs.filter { $0.outputDeleted == true }.map(\.id)) == deletedIDs, "Image batch did not persist every affected record")
        try check(persisted.jobs.count == 6, "Deleting images removed task records")
        try FileManager.default.removeItem(atPath: jobs[2].request.output)
        try imageStore.trashOutputs([jobs[2].id])
        try check(imageStore.jobs.first { $0.id == jobs[2].id }?.outputDeleted == true, "A missing output could not be removed from the gallery")
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 64,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.bitmapData!.initialize(repeating: 0, count: bitmap.bytesPerRow * 64)
        let png = bitmap.representation(using: .png, properties: [:])!
        for mode in ["before-rename", "after-rename", "corrupt", "missing-receipt"] {
            let folder = root.appendingPathComponent("recovery-\(mode)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            var request = NativeRequest(prompt: "recovery fixture", output: folder.appendingPathComponent("final.png").path)
            request.operation = "image.generate"; request.width = 64; request.height = 64
            var transaction: ImageOutputTransaction? = try ImageOutputTransaction(request: request)
            let staged = transaction!.stagedURL
            try png.write(to: staged)
            let payload: [String: Any] = ["schema_version": 1, "model": request.model,
                "operation": "image.generate", "output": staged.path, "width": 64, "height": 64,
                "steps": request.steps, "seed": request.seed, "warmup": false]
            let result = try transaction!.prepare(JSONSerialization.data(withJSONObject: payload))
            var pending = NativeJob(id: UUID(), createdAt: Date(), request: request,
                state: "finalizing", phase: "export", completed: 0, total: 1, elapsed: 1)
            if mode != "missing-receipt" { pending.resultJSON = String(decoding: result, as: UTF8.self) }
            try JSONEncoder().encode([pending]).write(to: folder.appendingPathComponent("jobs.json"), options: .atomic)
            transaction!.retainForRecovery()
            if mode == "after-rename" { try transaction!.publish() }
            transaction = nil
            if mode == "corrupt" { try Data("corrupt".utf8).write(to: staged) }
            let recovered = NativeJobStore(directory: folder)
            let expected = ["before-rename", "after-rename"].contains(mode) ? "succeeded" : "failed"
            try check(recovered.storageError == nil && recovered.jobs.first?.state == expected,
                      "Job recovery failed: \(mode)")
            if expected == "succeeded" {
                try check(try Data(contentsOf: URL(fileURLWithPath: request.output)) == png, "Recovered job pixels changed")
            }
            let replay = NativeJobStore(directory: folder)
            try check(replay.jobs.first?.id == pending.id && replay.jobs.first?.state == expected,
                      "Recovery was not persisted/idempotent: \(mode)")
        }
        for blockedSave in [false, true] {
            let folder = root.appendingPathComponent(blockedSave ? "submission-save-failure" : "submission-model-failure")
            let submitted = NativeJobStore(directory: folder)
            if blockedSave {
                try FileManager.default.createDirectory(at: folder.appendingPathComponent("jobs.json"), withIntermediateDirectories: false)
            }
            var request = NativeRequest(prompt: "persist before model inspection", output: folder.appendingPathComponent("outputs/result.png").path)
            request.model = "z-image-turbo"; request.width = 512; request.height = 512; request.steps = 9
            let intent = NativeRequestV2(legacy: request, targetBytes: 10 << 30)
            var failure: Error?
            do {
                _ = try await submitted.generate(modelURL: folder.appendingPathComponent("missing-model"),
                                                 request: request, streamingRequest: intent)
            } catch { failure = error }
            try check(failure != nil && !submitted.busy && !submitted.canUnload, "Failed submission retained active work")
            try check(submitted.jobs.count == 1 && submitted.jobs[0].state == "failed", "Failed public preflight lost its job")
            let job = submitted.jobs[0]
            try check(job.publicStreamingResolutionJSON == nil, "Failed preflight persisted an authority")
            let stored = try JSONDecoder().decode(NativeRequestV2.self, from: Data((job.publicStreamingIntentJSON ?? "").utf8))
            try check(stored.outputs[0].path == request.output && stored.execution.streaming == intent.execution.streaming,
                      "Original selector or output was changed before persistence")
            try check(!FileManager.default.fileExists(atPath: request.output), "Preflight failure published output")
            let leftovers = try FileManager.default.contentsOfDirectory(atPath: folder.appendingPathComponent("outputs").path)
            try check(!leftovers.contains { $0.hasPrefix(".tc-image-staging-") }, "Failed submission leaked staging")
            if blockedSave {
                try check(submitted.storageError != nil && !(failure?.localizedDescription.contains("Z-Image 模型检查失败") ?? false),
                          "Model inspection preceded the failed initial save")
            } else {
                let restored = NativeJobStore(directory: folder)
                try check(restored.jobs.first?.id == job.id && restored.jobs.first?.state == "failed" &&
                          restored.jobs.first?.publicStreamingIntentJSON == job.publicStreamingIntentJSON,
                          "Preflight failure and original intent did not survive reopening")
            }
        }
        try await PublicImageJobTests.run(root: root.appendingPathComponent("public-worker-tests"))
        print("PASS: history operations, hybrid precision, finalizing recovery, durable public submission and public worker jobs")
    }
}
