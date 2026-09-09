import Foundation
import Darwin

@main struct TensorCacheTests {
    static func main() throws {
        let fm = FileManager.default, root = fm.temporaryDirectory.appendingPathComponent("tc-tensor-cache-\(UUID())")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        func check(_ ok: Bool, _ text: String) throws { if !ok { throw LibraryFailure(message: text) } }
        let cacheRoot = root.appendingPathComponent("cache")
        try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cache = TensorCache(roots: [cacheRoot])
        let now = Date(), old = now.addingTimeInterval(-40 * 86400)
        func fixture(_ letter: String, date: Date) throws -> URL {
            let key = String(repeating: letter, count: 64)
            let entry = cacheRoot.appendingPathComponent(key), directory = entry.appendingPathComponent("conditioning")
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            let meta = try JSONSerialization.data(withJSONObject: ["format": "turbocider-ltx-conditioning-v1", "cache": true, "cache_identity": key])
            try meta.write(to: directory.appendingPathComponent("conditioning.json"))
            for name in ["video_context.bf16", "audio_context.bf16", "text_mask.bf16"] { try Data([1,2]).write(to: directory.appendingPathComponent(name)) }
            for file in try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
                try fm.setAttributes([.modificationDate: date], ofItemAtPath: file.path)
            }
            return entry
        }
        let expired = try fixture("a", date: old), fresh = try fixture("b", date: now)
        let unknown = try fixture("c", date: old)
        try Data([5]).write(to: unknown.appendingPathComponent("important.png"))
        let external = root.appendingPathComponent("external"); try fm.createDirectory(at: external, withIntermediateDirectories: true)
        try Data([9]).write(to: external.appendingPathComponent("model.safetensors"))
        let linked = cacheRoot.appendingPathComponent(String(repeating: "d", count: 64))
        try fm.createSymbolicLink(at: linked, withDestinationURL: external)
        let unsafe = try fixture("e", date: old)
        let tensor = unsafe.appendingPathComponent("conditioning/video_context.bf16")
        try fm.removeItem(at: tensor); try fm.createSymbolicLink(at: tensor, withDestinationURL: external.appendingPathComponent("model.safetensors"))
        let lockURL = root.appendingPathComponent("gpu.lock")
        let held = try LibraryLease(lockURL)
        do { _ = try cache.prune(olderThanDays: 0, now: now, leaseURL: lockURL); throw LibraryFailure(message: "Pruned active runtime") }
        catch { try check(error.localizedDescription.contains("推理"), "Wrong busy-cache error") }
        withExtendedLifetime(held) {}
        // A separate lease path represents an idle runtime for this fixture.
        let freeLock = root.appendingPathComponent("idle.lock")
        let report = try cache.prune(olderThanDays: 30, now: now, leaseURL: freeLock)
        try check(report.removedEntries == 1 && report.removedBytes > 0 && report.entries.count == 1, "Retention selected wrong entries")
        try check(!fm.fileExists(atPath: expired.path) && fm.fileExists(atPath: fresh.path), "Retention age boundary failed")
        let cleared = try cache.prune(olderThanDays: 0, now: now.addingTimeInterval(1), leaseURL: freeLock)
        try check(cleared.removedEntries == 1 && cleared.entries.isEmpty, "Manual cleanup failed")
        try check(fm.fileExists(atPath: unknown.appendingPathComponent("important.png").path) && fm.fileExists(atPath: linked.path) && fm.fileExists(atPath: unsafe.path), "Unknown or linked entry was deleted")
        try check(try Data(contentsOf: external.appendingPathComponent("model.safetensors")) == Data([9]), "Cleanup altered external weights")
        let dumps = root.appendingPathComponent("outputs/dumps")
        try fm.createDirectory(at: dumps.appendingPathComponent("nested"), withIntermediateDirectories: true)
        func dump(_ name: String, key: String = "tensor", date: Date = old) throws -> URL {
            let file = dumps.appendingPathComponent(name)
            let header = try JSONSerialization.data(withJSONObject: [key: ["dtype": "BF16", "shape": [1], "data_offsets": [0,2]]])
            var length = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(header); data.append(contentsOf: [0,0])
            try data.write(to: file); try fm.setAttributes([.modificationDate: date], ofItemAtPath: file.path)
            return file
        }
        let oldDump = try dump("z_latent_step_1.safetensors"), freshDump = try dump("latent_0.safetensors", date: now)
        let modelWeights = try dump("model.safetensors"), disguisedWeights = try dump("conditioning.safetensors", key: "model.weight")
        let nested = try dump("nested/z_decoded.safetensors")
        let linkedDump = dumps.appendingPathComponent("z_latent_final.safetensors")
        try fm.createSymbolicLink(at: linkedDump, withDestinationURL: oldDump)
        try Data([7]).write(to: dumps.appendingPathComponent("result.png"))
        let malformed = dumps.appendingPathComponent("z_decoded.safetensors")
        try Data([1,2,3]).write(to: malformed)
        let registryFile = root.appendingPathComponent("registry.json")
        setenv("TURBOCIDER_DUMP_CACHE_SETTINGS", registryFile.path, 1)
        defer { unsetenv("TURBOCIDER_DUMP_CACHE_SETTINGS") }
        let enrolled = try DiagnosticTensorRegistry.update(dumps, remove: false)
        try check(enrolled.directories == [dumps.path], "Diagnostic directory registration failed")
        let diagnosticCache = TensorCache(roots: [], diagnosticDirectories: [dumps])
        let dumpInventory = try diagnosticCache.inventory()
        try check(dumpInventory.entries.count == 2, "Unsafe or nested diagnostic file included")
        let prunedDumps = try diagnosticCache.prune(olderThanDays: 30, now: now, leaseURL: freeLock)
        try check(prunedDumps.removedEntries == 1 && !fm.fileExists(atPath: oldDump.path) && fm.fileExists(atPath: freshDump.path), "Diagnostic retention failed")
        for file in [modelWeights, disguisedWeights, nested, malformed, dumps.appendingPathComponent("result.png")] {
            try check(fm.fileExists(atPath: file.path), "Diagnostic cleanup deleted protected file: \(file.lastPathComponent)")
        }
        let forgotten = try DiagnosticTensorRegistry.update(dumps, remove: true)
        try check(forgotten.directories.isEmpty && fm.fileExists(atPath: freshDump.path), "Forgetting deleted files")
        print("PASS: tensor retention, manual cleanup, active-runtime exclusion, unknown/media/model preservation, symlink safety, explicit diagnostic registration, nonrecursive age cleanup and forget-without-deletion")
    }
}
