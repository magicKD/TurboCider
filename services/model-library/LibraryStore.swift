import Foundation
import CryptoKit
import Darwin

struct LibraryFailure: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct LibraryComponent: Codable, Sendable, Equatable {
    var path: String
    /// Semantic identity is explicit: a directory name is not compatibility proof.
    var compatibility: String
}

struct LibraryInstallation: Codable, Identifiable, Sendable {
    var id: String
    var modelID: String
    var name: String
    var path: String
    var managed: Bool
    var provider: String?
    var repository: String?
    var revision: String?
    var manifest: String?
    var components: [String: LibraryComponent]
    var createdAt: Date
}

struct LibraryLoRA: Codable, Identifiable, Sendable {
    var id: String
    var modelID: String
    var path: String
    var name: String
}

struct LibraryIndex: Codable, Sendable {
    var schemaVersion = 1
    var installations: [LibraryInstallation] = []
    var loras: [LibraryLoRA]? = nil
}

/// An OS lock, not a stale timestamp file. A crashed process releases the lease.
/// One writer is permitted across the App and all CLI processes.
final class LibraryLease: @unchecked Sendable {
    private let descriptor: Int32
    init(_ file: URL) throws {
        descriptor = Darwin.open(file.path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw LibraryFailure(message: "Cannot open model-library lock.") }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            Darwin.close(descriptor)
            throw LibraryFailure(message: "Model library is busy in another App or CLI operation.")
        }
    }
    deinit { flock(descriptor, LOCK_UN); Darwin.close(descriptor) }
}

/// The same Foundation implementation is compiled into the native library helper
/// used by App and CLI. It does not depend on Python, MLX or a visible GPU.
struct LibraryStore: Sendable {
    let root: URL
    private var indexURL: URL { root.appendingPathComponent("library.json") }

    static var defaultRoot: URL {
        if let path = ProcessInfo.processInfo.environment["TURBOCIDER_MODEL_LIBRARY"], !path.isEmpty {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        if let data = try? Data(contentsOf: LibrarySettings.file),
           let settings = try? JSONDecoder().decode(LibrarySettings.self, from: data), settings.modelRoot.hasPrefix("/") {
            return URL(fileURLWithPath: settings.modelRoot, isDirectory: true)
        }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TurboCider/Models", isDirectory: true)
    }

    init(root: URL = LibraryStore.defaultRoot) throws {
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: self.root, withIntermediateDirectories: true)
    }

    func acquireLease() throws -> LibraryLease {
        try LibraryLease(root.appendingPathComponent(".library.lock"))
    }

    func read() throws -> LibraryIndex {
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return LibraryIndex() }
        let value = try JSONDecoder().decode(LibraryIndex.self, from: Data(contentsOf: indexURL))
        guard value.schemaVersion == 1 else { throw LibraryFailure(message: "Unsupported model-library schema.") }
        guard Set(value.installations.map(\.id)).count == value.installations.count else {
            throw LibraryFailure(message: "Duplicate installation IDs in model library.")
        }
        return value
    }

    private func write(_ index: LibraryIndex) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(index).write(to: indexURL, options: .atomic)
    }

    /// Record an existing directory. Registration never copies or removes it.
    @discardableResult func register(modelID: String, path: URL, name: String? = nil,
                                    components: [String: LibraryComponent] = [:]) throws -> LibraryInstallation {
        let lease = try acquireLease(); defer { withExtendedLifetime(lease) {} }
        try Self.validateIdentifier(modelID)
        let canonical = path.standardizedFileURL.resolvingSymlinksInPath()
        guard try canonical.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
            throw LibraryFailure(message: "Choose an existing model directory.")
        }
        for (key, component) in components {
            try Self.validateRelativePath(key)
            guard !component.compatibility.isEmpty, FileManager.default.fileExists(atPath: component.path) else {
                throw LibraryFailure(message: "Shared component is missing or has no compatibility identity: \(key)")
            }
        }
        var index = try read()
        if let existing = index.installations.first(where: { $0.modelID == modelID && $0.path == canonical.path }) { return existing }
        let installation = LibraryInstallation(id: UUID().uuidString, modelID: modelID,
            name: name ?? canonical.lastPathComponent, path: canonical.path, managed: false,
            components: components, createdAt: Date())
        index.installations.append(installation); try write(index)
        return installation
    }

    @discardableResult func registerLoRA(modelID: String, path: URL) throws -> LibraryLoRA {
        let lease = try acquireLease(); defer { withExtendedLifetime(lease) {} }
        try Self.validateIdentifier(modelID)
        let canonical = path.standardizedFileURL.resolvingSymlinksInPath()
        guard canonical.pathExtension == "safetensors",
              try canonical.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
            throw LibraryFailure(message: "请选择存在的 .safetensors LoRA 文件。")
        }
        var index = try read()
        if let existing = index.loras?.first(where: { $0.modelID == modelID && $0.path == canonical.path }) { return existing }
        let item = LibraryLoRA(id: UUID().uuidString, modelID: modelID, path: canonical.path, name: canonical.lastPathComponent)
        index.loras = (index.loras ?? []) + [item]
        try write(index)
        return item
    }

    func unregisterLoRA(id: String) throws {
        let lease = try acquireLease(); defer { withExtendedLifetime(lease) {} }
        var index = try read()
        index.loras = (index.loras ?? []).filter { $0.id != id }
        try write(index)
    }

    /// Remove only the registration. Both external and managed bytes are retained.
    func unregister(id: String) throws {
        let lease = try acquireLease(); defer { withExtendedLifetime(lease) {} }
        var index = try read()
        guard index.installations.contains(where: { $0.id == id }) else { throw LibraryFailure(message: "Unknown installation.") }
        index.installations.removeAll { $0.id == id }; try write(index)
    }

    /// Called while the caller holds a lease, after every artifact is verified.
    /// Staging is private to the library; atomic rename publishes complete models.
    func publish(staging: URL, modelID: String, name: String, provider: String,
                 repository: String, revision: String,
                 components: [String: LibraryComponent], manifest: String? = nil) throws -> LibraryInstallation {
        try Self.validateIdentifier(modelID)
        guard staging.deletingLastPathComponent().standardizedFileURL.path == root.appendingPathComponent("staging").standardizedFileURL.path,
              staging.resolvingSymlinksInPath().path == staging.standardizedFileURL.path,
              try staging.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
            throw LibraryFailure(message: "Installation staging must belong to the model library.")
        }
        var index = try read()
        let id = UUID().uuidString
        let parent = try managedDirectory("installations")
        let destination = parent.appendingPathComponent(id)
        try FileManager.default.moveItem(at: staging, to: destination)
        let installation = LibraryInstallation(id: id, modelID: modelID, name: name,
            path: destination.path, managed: true, provider: provider, repository: repository,
            revision: revision, manifest: manifest, components: components, createdAt: Date())
        index.installations.append(installation)
        do { try write(index) }
        catch {
            // Roll back only this operation's new directory, never another model.
            try? FileManager.default.moveItem(at: destination, to: staging)
            throw error
        }
        return installation
    }

    func managedDirectory(_ name: String) throws -> URL {
        try Self.validateIdentifier(name)
        let value = root.appendingPathComponent(name, isDirectory: true)
        if FileManager.default.fileExists(atPath: value.path) {
            guard value.resolvingSymlinksInPath().standardizedFileURL.path == value.standardizedFileURL.path else {
                throw LibraryFailure(message: "Managed library directories cannot be symlinks: \(name)")
            }
        }
        try FileManager.default.createDirectory(at: value, withIntermediateDirectories: true)
        return value
    }

    static func validateIdentifier(_ value: String) throws {
        guard !value.isEmpty, value.count <= 160,
              value.range(of: "^[A-Za-z0-9][A-Za-z0-9._-]*$", options: .regularExpression) != nil,
              value != ".", value != ".." else { throw LibraryFailure(message: "Invalid library identifier.") }
    }

    static func validateRelativePath(_ value: String) throws {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty, value.utf8.count < 4096, !value.contains("\\"),
              !value.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
              parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw LibraryFailure(message: "Unsafe repository-relative path.")
        }
    }

    /// Streaming SHA-256 avoids mapping multi-gigabyte weight files into memory.
    static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Commit downloaded bytes to the content store; link snapshots to one copy.
    /// The caller must hold a lease. Known hashes allow reuse across providers.
    func commitBlob(temporary: URL, expectedSize: Int64, expectedSHA256: String?) throws -> URL {
        let size = try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1
        guard expectedSize >= 0, size == expectedSize else { throw LibraryFailure(message: "Downloaded file size does not match repository metadata.") }
        let hash = try Self.sha256(temporary)
        if let expectedSHA256, expectedSHA256.lowercased() != hash {
            throw LibraryFailure(message: "Downloaded file failed SHA-256 verification.")
        }
        let destination = try managedDirectory("blobs").appendingPathComponent(hash)
        if FileManager.default.fileExists(atPath: destination.path) {
            guard destination.resolvingSymlinksInPath().path == destination.path,
                  try Self.sha256(destination) == hash else {
                throw LibraryFailure(message: "Existing content-store blob is corrupt.")
            }
            try FileManager.default.removeItem(at: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: destination)
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: destination.path)
        }
        return destination
    }

    /// Link a validated blob or an explicitly selected compatible local component.
    /// Reject ancestor symlinks, including collisions with an earlier component.
    static func link(_ source: URL, at relative: String, in staging: URL) throws {
        try validateRelativePath(relative)
        let destination = staging.appendingPathComponent(relative)
        var ancestor = destination.deletingLastPathComponent()
        while ancestor.path != staging.path {
            guard ancestor.path.hasPrefix(staging.path + "/"), ancestor.resolvingSymlinksInPath().path == ancestor.path else {
                throw LibraryFailure(message: "Artifact path crosses a component link.")
            }
            ancestor.deleteLastPathComponent()
        }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard FileManager.default.fileExists(atPath: source.path) else { throw LibraryFailure(message: "Shared artifact no longer exists.") }
        try FileManager.default.createSymbolicLink(at: destination, withDestinationURL: source.resolvingSymlinksInPath())
    }
}
