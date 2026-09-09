import Foundation
import CoreFoundation

struct DiagnosticTensorRegistry: Codable {
    var directories: [String] = []
    static var file: URL {
        if let path = ProcessInfo.processInfo.environment["TURBOCIDER_DUMP_CACHE_SETTINGS"] { return URL(fileURLWithPath: path) }
        return TensorCacheSettings.file.deletingLastPathComponent().appendingPathComponent("diagnostic-tensor-directories.json")
    }
    static func read() throws -> Self {
        guard FileManager.default.fileExists(atPath: file.path) else { return Self() }
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: file))
        guard value.directories.allSatisfy({ $0.hasPrefix("/") && $0 != "/" }) else { throw LibraryFailure(message: "Invalid diagnostic tensor directory registry.") }
        return value
    }
    static func update(_ root: URL, remove: Bool) throws -> Self {
        let fm = FileManager.default
        try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let lease = try LibraryLease(file.appendingPathExtension("lock")); defer { withExtendedLifetime(lease) {} }
        var value = try read()
        let path = root.standardizedFileURL.resolvingSymlinksInPath().path
        if remove { value.directories.removeAll { $0 == path || $0 == root.standardizedFileURL.path } }
        else {
            guard path != "/", try fm.contentsOfDirectory(at: URL(fileURLWithPath: path), includingPropertiesForKeys: nil).contains(where: { (try? DiagnosticTensorCache.entry($0)) != nil }) else {
                throw LibraryFailure(message: "此目录没有可识别的 TurboCider 输出张量；请选择实际 dump 目录。不会递归登记整个 outputs。")
            }
            if !value.directories.contains(path) { value.directories.append(path) }
        }
        try JSONEncoder().encode(value).write(to: file, options: .atomic)
        return value
    }
}

enum DiagnosticTensorCache {
    /// Match public FLUX / Z-Image dump filenames AND the single `tensor`
    /// safetensors payload. Arbitrary checkpoint names/keys are excluded.
    static func entry(_ file: URL) throws -> TensorCacheEntry? {
        guard file.pathExtension == "safetensors" else { return nil }
        let name = file.deletingPathExtension().lastPathComponent
        let exact: Set<String> = ["conditioning", "initial_latent", "conditioned_initial_latent", "pixels_nhwc", "z_latent_initial", "z_latent_final", "z_decoded"]
        let numbered = ["input_image_", "image_latent_", "noise_", "latent_", "z_latent_step_"].contains { prefix in
            guard name.hasPrefix(prefix) else { return false }
            let number = name.dropFirst(prefix.count)
            return !number.isEmpty && number.utf8.allSatisfy { (48...57).contains($0) }
        }
        guard exact.contains(name) || numbered else { return nil }
        let value = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey])
        guard value.isRegularFile == true, value.isSymbolicLink == false, let size = value.fileSize, size > 8, let date = value.contentModificationDate else { return nil }
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else { return nil }
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        guard length > 0, length <= 1_048_576, length <= UInt64(size) - 8,
              let data = try handle.read(upToCount: Int(length)), data.count == Int(length),
              let header = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(header.keys).subtracting(["__metadata__"]) == ["tensor"],
              let tensor = header["tensor"] as? [String: Any], let dtype = tensor["dtype"] as? String,
              let width = ["BF16": UInt64(2), "F16": 2, "F32": 4][dtype],
              let shape = tensor["shape"] as? [Any], let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2 else { return nil }
        func integer(_ raw: Any) -> UInt64? {
            guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
            return UInt64(n.stringValue)
        }
        var bytes = width
        for raw in shape {
            guard let dim = integer(raw) else { return nil }
            let result = bytes.multipliedReportingOverflow(by: dim)
            guard !result.overflow else { return nil }; bytes = result.partialValue
        }
        guard integer(offsets[0]) == 0, integer(offsets[1]) == bytes, bytes == UInt64(size) - 8 - length else { return nil }
        return TensorCacheEntry(path: file.path, bytes: Int64(size), modifiedAt: date, category: "diagnostic_tensor")
    }
}
