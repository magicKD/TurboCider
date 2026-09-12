import Foundation
import Darwin

struct TensorCacheSettings: Codable, Sendable {
    var retentionDays = 0
    static var file: URL {
        if let path = ProcessInfo.processInfo.environment["TURBOCIDER_CACHE_SETTINGS"] { return URL(fileURLWithPath: path) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCider/cache-settings.json")
    }
    static func read() throws -> Self {
        guard FileManager.default.fileExists(atPath: file.path) else { return Self(retentionDays: 0) }
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: file))
        guard (0...3650).contains(value.retentionDays) else { throw LibraryFailure(message: "Invalid tensor cache retention.") }
        return value
    }
    func save() throws {
        guard (0...3650).contains(retentionDays) else { throw LibraryFailure(message: "Retention must be 0 (keep) or 1...3650 days.") }
        try FileManager.default.createDirectory(at: Self.file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let lease = try LibraryLease(Self.file.appendingPathExtension("lock")); defer { withExtendedLifetime(lease) {} }
        try JSONEncoder().encode(self).write(to: Self.file, options: .atomic)
    }
}

struct TensorCacheEntry: Codable, Sendable, Identifiable {
    var id: String { path }
    var path: String
    var bytes: Int64
    var modifiedAt: Date
    var category: String? = nil
}
struct TensorCacheReport: Codable, Sendable {
    var roots: [String]
    var entries: [TensorCacheEntry]
    var skipped: Int
    var removedBytes: Int64 = 0
    var removedEntries: Int = 0
    var bytes: Int64 { entries.reduce(0) { $0 + $1.bytes } }
}

/// Prune native conditioning and explicitly enrolled diagnostic tensors only.
struct TensorCache {
    let roots: [URL]
    var diagnosticDirectories: [URL] = []
    static var sharedRoot: URL {
        if let path = ProcessInfo.processInfo.environment["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"], !path.isEmpty { return URL(fileURLWithPath: path) }
        return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCider/ltx-conditioning")
    }
    static var standard: Self {
        let state = ProcessInfo.processInfo.environment["TURBOCIDER_NATIVE_STATE"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative")
        let directories = (try? DiagnosticTensorRegistry.read().directories) ?? []
        return Self(roots: [sharedRoot, state.appendingPathComponent("api-service/jobs/ltx-conditioning-cache")],
                    diagnosticDirectories: directories.map { URL(fileURLWithPath: $0) })
    }
    private func entry(_ folder: URL) throws -> TensorCacheEntry? {
        let fm = FileManager.default, key = folder.lastPathComponent
        guard key.count == 64, key.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { return nil }
        func plainDirectory(_ url: URL) throws -> Bool {
            let value = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
            return value.isSymbolicLink == false && value.isDirectory == true
        }
        guard try plainDirectory(folder) else { return nil }
        let top = try fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        guard top.count == 1, top[0].lastPathComponent == "conditioning", try plainDirectory(top[0]) else { return nil }
        let files = try fm.contentsOfDirectory(at: top[0], includingPropertiesForKeys: nil)
        let expected: Set<String> = ["conditioning.json", "video_context.bf16", "audio_context.bf16", "text_mask.bf16"]
        guard Set(files.map(\.lastPathComponent)) == expected else { return nil }
        var bytes: Int64 = 0; var modified = Date.distantPast
        for file in files {
            let value = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey])
            guard value.isRegularFile == true, value.isSymbolicLink == false, let size = value.fileSize, size >= 0,
                  let date = value.contentModificationDate else { return nil }
            if file.lastPathComponent == "conditioning.json", size > 1_048_576 { return nil }
            bytes += Int64(size); modified = max(modified, date)
        }
        let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: top[0].appendingPathComponent("conditioning.json"))) as? [String: Any]
        guard metadata?["format"] as? String == "turbocider-ltx-conditioning-v1", metadata?["cache"] as? Bool == true,
              metadata?["cache_identity"] as? String == key else { return nil }
        return TensorCacheEntry(path: folder.path, bytes: bytes, modifiedAt: modified)
    }
    func inventory() throws -> TensorCacheReport {
        let fm = FileManager.default
        var report = TensorCacheReport(roots: (roots + diagnosticDirectories).map(\.path), entries: [], skipped: 0)
        var seen: Set<String> = []
        for root in roots {
            guard fm.fileExists(atPath: root.path) else { continue }
            // Canonicalize the configured root once; never follow per-entry links.
            let canonical = root.resolvingSymlinksInPath()
            guard seen.insert(canonical.path).inserted else { continue }
            for folder in try fm.contentsOfDirectory(at: canonical, includingPropertiesForKeys: nil) {
                try Task.checkCancellation()
                do { if let value = try entry(folder) { report.entries.append(value) } else { report.skipped += 1 } }
                catch is CancellationError { throw CancellationError() }
                catch { report.skipped += 1 }
            }
        }
        for root in diagnosticDirectories {
            guard fm.fileExists(atPath: root.path) else { continue }
            // Registration records canonical directories. A replacement symlink
            // must not silently enroll files from a different output directory.
            guard root.resolvingSymlinksInPath().path == root.standardizedFileURL.path else { report.skipped += 1; continue }
            for file in try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
                try Task.checkCancellation()
                do {
                    if let value = try DiagnosticTensorCache.entry(file), !report.entries.contains(where: { $0.path == value.path }) { report.entries.append(value) }
                    else { report.skipped += 1 }
                } catch { report.skipped += 1 }
            }
        }
        report.entries.sort { $0.modifiedAt < $1.modifiedAt }; return report
    }
    func prune(olderThanDays days: Int, now: Date = Date(), leaseURL: URL? = nil) throws -> TensorCacheReport {
        guard (0...3650).contains(days) else { throw LibraryFailure(message: "Invalid age threshold.") }
        let path = leaseURL ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("turbocider-gpu-\(geteuid()).lock")
        let lease: LibraryLease
        do { lease = try LibraryLease(path) } catch { throw LibraryFailure(message: "推理正在运行或无法取得缓存维护锁，请在空闲时重试。") }
        defer { withExtendedLifetime(lease) {} }
        var report = try inventory()
        let cutoff = now.addingTimeInterval(-Double(days) * 86400)
        for candidate in report.entries where candidate.modifiedAt <= cutoff {
            try Task.checkCancellation()
            // Revalidate just before removal while native writers are excluded.
            let folder = URL(fileURLWithPath: candidate.path)
            if candidate.category == "diagnostic_tensor", folder.deletingLastPathComponent().resolvingSymlinksInPath().path != folder.deletingLastPathComponent().standardizedFileURL.path { continue }
            let current = candidate.category == "diagnostic_tensor" ? try DiagnosticTensorCache.entry(folder) : try entry(folder)
            guard let current, current.modifiedAt <= cutoff else { continue }
            try FileManager.default.removeItem(at: folder)
            report.removedEntries += 1; report.removedBytes += current.bytes
        }
        let fresh = try inventory(); report.entries = fresh.entries; report.skipped = fresh.skipped
        return report
    }
}
