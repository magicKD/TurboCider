import Foundation

struct LibrarySupplement: Codable, Sendable {
    var repository: String
    var revision: String? = nil
    /// Explicit paths only; supplementary repositories are never downloaded whole.
    var include: [String]
}

struct LibraryDownloadRequest: Codable, Sendable {
    var modelID: String
    var repository: String
    var provider: HubProvider = .modelscope
    var revision: String?
    var name: String?
    /// Exact paths or directory prefixes ending in '/'. Empty means all files.
    var include: [String] = []
    var components: [String: LibraryComponent] = [:]
    var supplements: [LibrarySupplement] = []
    init(modelID: String, repository: String, provider: HubProvider = .modelscope,
         revision: String? = nil, name: String? = nil, include: [String] = [], components: [String: LibraryComponent] = [:], supplements: [LibrarySupplement] = []) {
        self.modelID = modelID; self.repository = repository; self.provider = provider
        self.revision = revision; self.name = name; self.include = include; self.components = components
        self.supplements = supplements
    }
    private enum CodingKeys: String, CodingKey { case modelID, repository, provider, revision, name, include, components, supplements }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decode(String.self, forKey: .modelID)
        repository = try c.decode(String.self, forKey: .repository)
        provider = try c.decodeIfPresent(HubProvider.self, forKey: .provider) ?? .modelscope
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        name = try c.decodeIfPresent(String.self, forKey: .name)
        include = try c.decodeIfPresent([String].self, forKey: .include) ?? []
        components = try c.decodeIfPresent([String: LibraryComponent].self, forKey: .components) ?? [:]
        supplements = try c.decodeIfPresent([LibrarySupplement].self, forKey: .supplements) ?? []
    }
}

struct LibraryDownloadPlan: Codable, Sendable {
    var snapshot: HubSnapshot
    var files: [HubFile]
    var reusedComponents: [String: LibraryComponent]
    var downloadBytes: Int64
    var cachedBytes: Int64
    var additionalSnapshots: [HubSnapshot]? = nil
}

struct LibraryDownloadEvent: Codable, Sendable {
    var phase: String
    var file: String
    var completedFiles: Int
    var totalFiles: Int
    var completedBytes: Int64
    var totalBytes: Int64
}

struct LibraryDownloader: Sendable {
    let store: LibraryStore
    let client: HubClient

    func plan(_ request: LibraryDownloadRequest) async throws -> LibraryDownloadPlan {
        try LibraryStore.validateIdentifier(request.modelID)
        guard request.provider == client.provider else { throw LibraryFailure(message: "Hub provider does not match request.") }
        for prefix in request.include {
            try LibraryStore.validateRelativePath(prefix.hasSuffix("/") ? String(prefix.dropLast()) : prefix)
        }
        for (path, component) in request.components {
            try LibraryStore.validateRelativePath(path)
            guard !component.compatibility.isEmpty,
                  FileManager.default.fileExists(atPath: component.path) else {
                throw LibraryFailure(message: "Selected shared component is missing or unidentified: \(path)")
            }
        }
        if request.modelID == "z-image-turbo", !request.components.isEmpty {
            guard let encoder = request.components["text_encoder"], let tokenizer = request.components["tokenizer"] else {
                throw LibraryFailure(message: "Z-Image 的共享文本编码器和 tokenizer 必须成对选择。")
            }
            let inspected = try SharedTextComponents.inspect(URL(fileURLWithPath: encoder.path).deletingLastPathComponent())
            guard inspected["text_encoder"]?.path == URL(fileURLWithPath: encoder.path).resolvingSymlinksInPath().path,
                  inspected["tokenizer"]?.path == URL(fileURLWithPath: tokenizer.path).resolvingSymlinksInPath().path else {
                throw LibraryFailure(message: "共享 tokenizer 与文本编码器不属于同一组已检查组件。")
            }
        }
        let snapshot = try await client.snapshot(repository: request.repository, revision: request.revision)
        guard request.supplements.count <= 8 else { throw LibraryFailure(message: "At most eight supplementary repositories are supported.") }
        var allFiles = snapshot.files
        var additional: [HubSnapshot] = []
        for supplement in request.supplements {
            guard !supplement.include.isEmpty else { throw LibraryFailure(message: "Supplementary repositories require an explicit file selection.") }
            for path in supplement.include { try LibraryStore.validateRelativePath(path) }
            // The same provider/client enforces the same credential and redirect rules.
            let source = try await client.snapshot(repository: supplement.repository, revision: supplement.revision)
            for path in supplement.include {
                guard let original = source.files.first(where: { $0.path == path }) else {
                    throw LibraryFailure(message: "Required supplementary file is missing: \(supplement.repository)/\(path)")
                }
                var file = original; file.sourceRepository = supplement.repository; allFiles.append(file)
            }
            additional.append(source)
        }
        let selected = allFiles.filter { file in
            (request.include.isEmpty || request.include.contains { $0 == file.path || ($0.hasSuffix("/") && file.path.hasPrefix($0)) }) &&
            !(request.modelID == "z-image-turbo" && request.components["text_encoder"] != nil && file.path.hasPrefix("split_files/text_encoders/")) &&
            !request.components.keys.contains { file.path == $0 || file.path.hasPrefix($0 + "/") }
        }
        guard Set(selected.map(\.path)).count == selected.count else {
            throw LibraryFailure(message: "Repositories contain conflicting selected paths; select one source for each file.")
        }
        guard !selected.isEmpty || !request.components.isEmpty else { throw LibraryFailure(message: "No files match this download selection.") }
        for prefix in request.include {
            guard allFiles.contains(where: { $0.path == prefix || (prefix.hasSuffix("/") && $0.path.hasPrefix(prefix)) }) || request.components.keys.contains(where: { prefix == $0 || prefix == $0 + "/" }) else {
                throw LibraryFailure(message: "Required repository path is missing: \(prefix)")
            }
        }
        var cached: Int64 = 0, missing: Int64 = 0
        for file in selected {
            try Task.checkCancellation()
            if try cachedBlob(file) != nil { cached += file.size } else { missing += file.size }
        }
        return LibraryDownloadPlan(snapshot: snapshot, files: selected, reusedComponents: request.components,
                                   downloadBytes: missing, cachedBytes: cached, additionalSnapshots: additional.isEmpty ? nil : additional)
    }

    private func cachedBlob(_ file: HubFile) throws -> URL? {
        guard let sha = file.sha256 else { return nil }
        let blob = store.root.appendingPathComponent("blobs").appendingPathComponent(sha.lowercased())
        guard FileManager.default.fileExists(atPath: blob.path), blob.resolvingSymlinksInPath().path == blob.path else { return nil }
        let size = try blob.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
        guard size == file.size else { return nil }
        return try LibraryStore.sha256(blob) == sha.lowercased() ? blob : nil
    }

    func install(_ request: LibraryDownloadRequest,
                 progress: @escaping @Sendable (LibraryDownloadEvent) -> Void = { _ in }) async throws -> LibraryInstallation {
        let lease = try store.acquireLease(); defer { withExtendedLifetime(lease) {} }
        progress(LibraryDownloadEvent(phase: "resolving", file: "", completedFiles: 0, totalFiles: 0, completedBytes: 0, totalBytes: 0))
        let plan = try await plan(request)
        let capacity = try FileManager.default.attributesOfFileSystem(forPath: store.root.path)[.systemFreeSize] as? NSNumber
        guard let capacity, capacity.int64Value > plan.downloadBytes,
              capacity.int64Value - plan.downloadBytes > min(plan.downloadBytes / 20, 1_073_741_824) else {
            throw LibraryFailure(message: "Not enough free space for the selected model files.")
        }
        let staging = try store.managedDirectory("staging").appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        // Cancellation or failure leaves completed content-addressed blobs reusable,
        // but never publishes a partially installed model.
        defer { try? FileManager.default.removeItem(at: staging) }
        for (path, component) in request.components {
            try LibraryStore.link(URL(fileURLWithPath: component.path), at: path, in: staging)
        }
        let totalBytes = plan.files.reduce(Int64(0)) { $0 + $1.size }
        var completed: Int64 = 0
        for (index, file) in plan.files.enumerated() {
            try Task.checkCancellation()
            let blob: URL
            if let existing = try cachedBlob(file) {
                blob = existing
            } else {
                let temporary = staging.appendingPathComponent(".download-\(UUID())")
                let previous = completed
                try await client.download(file, repository: file.sourceRepository ?? request.repository, destination: temporary) { bytes, _ in
                    progress(LibraryDownloadEvent(phase: "downloading", file: file.path, completedFiles: index,
                        totalFiles: plan.files.count, completedBytes: previous + min(bytes, file.size), totalBytes: totalBytes))
                }
                progress(LibraryDownloadEvent(phase: "verifying", file: file.path, completedFiles: index,
                    totalFiles: plan.files.count, completedBytes: completed + file.size, totalBytes: totalBytes))
                blob = try store.commitBlob(temporary: temporary, expectedSize: file.size, expectedSHA256: file.sha256)
            }
            try LibraryStore.link(blob, at: file.path, in: staging)
            completed += file.size
            progress(LibraryDownloadEvent(phase: "installed_file", file: file.path, completedFiles: index + 1,
                totalFiles: plan.files.count, completedBytes: completed, totalBytes: totalBytes))
        }
        try Task.checkCancellation()
        let record = try JSONEncoder().encode(plan)
        // Reserved metadata lives in the index, not a potentially colliding upstream path.
        let manifests = try store.managedDirectory("manifests")
        let recordURL = manifests.appendingPathComponent("\(UUID()).json")
        try record.write(to: recordURL, options: .atomic)
        do {
            let result = try store.publish(staging: staging, modelID: request.modelID,
                name: request.name ?? request.repository, provider: request.provider.rawValue,
                repository: request.repository, revision: plan.snapshot.revision, components: request.components, manifest: recordURL.path)
            progress(LibraryDownloadEvent(phase: "complete", file: result.path, completedFiles: plan.files.count,
                totalFiles: plan.files.count, completedBytes: totalBytes, totalBytes: totalBytes))
            return result
        } catch { try? FileManager.default.removeItem(at: recordURL); throw error }
    }
}
