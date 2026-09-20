import Foundation

@main
struct HistoryManagementTests {
    @MainActor static func main() throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try condition() else { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected validation failure")
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
        print("PASS: modifier selection, batch history delete/undo/clear, atomic-save rollback, managed-file validation and batch trash")
    }
}
