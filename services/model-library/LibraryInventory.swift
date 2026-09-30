import Foundation
import Darwin

/// Metadata discovery shares the App's typed registry, but never creates a
/// library directory, reads model weights, or walks installation directories.
struct LibraryInventory: Encodable, Sendable {
    let schema_version = 1
    let root: String
    let index: LibraryIndex
    let scope = "registered_metadata"
    let files_verified = false

    static func read(root override: URL? = nil) throws -> Self {
        let root = try override ?? discoveryRoot()
        let store = try LibraryStore(root: root, createIfMissing: false)
        let data = try LibraryMetadataFile.read(store.root.appendingPathComponent("library.json"), limit: 4 * 1024 * 1024)
        let index = try data.map { try JSONDecoder().decode(LibraryIndex.self, from: $0) } ?? LibraryIndex()
        guard index.schemaVersion == 1 else { throw LibraryFailure(message: "Unsupported model-library schema.") }
        try unique(index.installations.map(\.id), kind: "installation")
        try unique((index.loras ?? []).map(\.id), kind: "LoRA")
        try unique((index.anePartitions ?? []).map(\.id), kind: "ANE partition")
        for item in index.installations {
            try absolute(item.path, field: "installation path")
            for component in item.components.values { try absolute(component.path, field: "component path") }
        }
        for item in index.loras ?? [] { try absolute(item.path, field: "LoRA path") }
        for item in index.anePartitions ?? [] {
            try absolute(item.path, field: "ANE manifest path")
            try absolute(item.checkpoint, field: "ANE checkpoint path")
            if let path = item.sourceManifest { try absolute(path, field: "ANE source manifest path") }
            for adapter in item.loras { try absolute(adapter.path, field: "ANE adapter path") }
        }
        return Self(root: store.root.path, index: index)
    }

    private static func unique(_ ids: [String], kind: String) throws {
        guard !ids.contains(where: { $0.isEmpty }), Set(ids).count == ids.count else {
            throw LibraryFailure(message: "Empty or duplicate \(kind) IDs in model library.")
        }
    }
    private static func absolute(_ path: String, field: String) throws {
        guard path.hasPrefix("/"), !path.contains("\0") else {
            throw LibraryFailure(message: "Model library \(field) must be an absolute local path.")
        }
    }

    private static func discoveryRoot() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        if let path = environment["TURBOCIDER_MODEL_LIBRARY"], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        if let data = try LibraryMetadataFile.read(LibrarySettings.file, limit: 1024 * 1024) {
            let settings = try JSONDecoder().decode(LibrarySettings.self, from: data)
            try absolute(settings.modelRoot, field: "configured root")
            return URL(fileURLWithPath: settings.modelRoot, isDirectory: true)
        }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TurboCider/Models", isDirectory: true)
    }
}

private enum LibraryMetadataFile {
    static func read(_ url: URL, limit: Int) throws -> Data? {
        // Nonblocking open lets us reject a FIFO/device in place of metadata
        // without waiting for a writer. A missing index is an empty registry;
        // access errors and malformed contents are never reported as empty.
        let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
        guard descriptor >= 0 else {
            if errno == ENOENT { return nil }
            throw LibraryFailure(message: "Cannot open library metadata \(url.lastPathComponent): \(String(cString: strerror(errno)))")
        }
        defer { Darwin.close(descriptor) }
        var attributes = stat()
        guard fstat(descriptor, &attributes) == 0, attributes.st_mode & S_IFMT == S_IFREG else {
            throw LibraryFailure(message: "Library metadata must be a regular file: \(url.lastPathComponent)")
        }
        guard attributes.st_size <= limit else {
            throw LibraryFailure(message: "Library metadata exceeds the \(limit)-byte discovery limit.")
        }
        var data = Data(), buffer = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            try Task.checkCancellation()
            let count = buffer.withUnsafeMutableBytes { Darwin.read(descriptor, $0.baseAddress, $0.count) }
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw LibraryFailure(message: "Cannot read library metadata: \(String(cString: strerror(errno)))")
            }
            guard data.count <= limit - count else {
                throw LibraryFailure(message: "Library metadata exceeds the \(limit)-byte discovery limit.")
            }
            data.append(contentsOf: buffer.prefix(count))
        }
        return data
    }
}
