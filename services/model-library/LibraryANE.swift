import Foundation

/// Inventory metadata is a snapshot. Discovery and the engine re-read manifests
/// before use; registration never asserts hardware residency or tensor parity.
struct LibraryANEPartition: Codable, Identifiable, Sendable {
    var id: String
    var modelID: String
    var path: String
    var kind: String
    var inputMode: String
    var buckets: [Int]
    var checkpoint: String
    var sourceManifest: String?
    var loras: [Adapter]
    var partitionCount: Int
    struct Adapter: Codable, Sendable {
        var path: String
        var strength: Double
        var role: String
    }
    var capacity: String {
        inputMode == "fixed" ? "固定 \(buckets.first ?? 0) 行" : "\(inputMode == "range" ? "范围变长" : "枚举变长") \(buckets.first ?? 0)–\(buckets.last ?? 0) 行"
    }

    static func inspect(modelID: String, manifest: URL) throws -> Self {
        let file = manifest.standardizedFileURL.resolvingSymlinksInPath()
        guard file.pathExtension == "json",
              let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 8 * 1024 * 1024,
              let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
              raw["schema_version"] as? Int == 2,
              let shape = raw["shape"] as? [String: Any],
              let buckets = shape["buckets"] as? [Int], !buckets.isEmpty, buckets.count <= 128,
              buckets == Array(Set(buckets)).sorted(), buckets.allSatisfy({ (1...8192).contains($0) }),
              let source = raw["source"] as? [String: Any], let checkpoint = source["checkpoint"] as? String,
              checkpoint.hasPrefix("/"), (source["checkpoint_bytes"] as? UInt64 ?? 0) > 0,
              let artifacts = raw["artifacts"] as? [String: [String: String]] else {
            throw LibraryFailure(message: "ANE manifest 无效：需要 schema 2、形状、checkpoint 和完整分区列表。")
        }
        let hidden = shape["K"] as? Int
        let inferred = hidden == 3840 ? "z-image-turbo" : hidden == 3072 ? "flux2-klein-4b" : ""
        let model = modelID == "auto" ? inferred : modelID
        guard !inferred.isEmpty, model == inferred, shape["N"] as? Int == hidden else {
            throw LibraryFailure(message: "ANE 分区架构与所选基础模型不匹配。")
        }
        let mode = shape["input_mode"] as? String ?? "fixed"
        guard ["fixed", "range", "enumerated"].contains(mode), mode != "fixed" || buckets.count == 1 else {
            throw LibraryFailure(message: "ANE 形状模式与容量列表不一致。")
        }
        let count = model == "z-image-turbo" ? 32 : 20
        guard artifacts.count == count else { throw LibraryFailure(message: "ANE 分区不完整：需要 \(count) 个分区。") }
        var kind: String?
        for index in 0..<count {
            guard let relative = artifacts[String(index)]?["int8_pc"] else { throw LibraryFailure(message: "ANE 分区缺少 block \(index)。") }
            try LibraryStore.validateRelativePath(relative)
            let artifact = file.deletingLastPathComponent().appendingPathComponent(relative)
            let resolved = artifact.resolvingSymlinksInPath()
            guard resolved.path == artifact.path, resolved.path.hasPrefix(file.deletingLastPathComponent().path + "/"),
                  ["mlpackage", "mlmodelc"].contains(artifact.pathExtension),
                  try artifact.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
                throw LibraryFailure(message: "ANE 分区缺失、路径越界或类型无效：\(relative)")
            }
            let current = artifact.pathExtension == "mlpackage" ? "source" : "compiled"
            guard kind == nil || kind == current else { throw LibraryFailure(message: "不能混合源分区与编译分区。") }
            kind = current
        }
        var adapters: [Adapter] = []
        if let values = source["loras"] {
            guard let entries = values as? [[String: Any]] else { throw LibraryFailure(message: "ANE LoRA 身份无效。") }
            for entry in entries {
                guard let path = entry["path"] as? String, path.hasPrefix("/"),
                      let strength = entry["strength"] as? Double, strength.isFinite,
                      let role = entry["role"] as? String, role == "transformer",
                      let hash = entry["sha256"] as? String, hash.count == 64, hash.allSatisfy({ $0.isHexDigit }),
                      entry["bytes"] as? UInt64 != nil else { throw LibraryFailure(message: "ANE LoRA 缺少文件身份、角色或强度。") }
                adapters.append(Adapter(path: path, strength: strength, role: role))
            }
        }
        return Self(id: UUID().uuidString, modelID: model, path: file.path, kind: kind!, inputMode: mode,
                    buckets: buckets, checkpoint: checkpoint, sourceManifest: raw["source_manifest"] as? String,
                    loras: adapters, partitionCount: count)
    }

    /// Read only: do not create directories while resolving a generation request.
    static func registered(modelID: String, root: URL = LibraryStore.defaultRoot) -> [Self] {
        guard let data = try? Data(contentsOf: root.appendingPathComponent("library.json")),
              let index = try? JSONDecoder().decode(LibraryIndex.self, from: data), index.schemaVersion == 1 else { return [] }
        return (index.anePartitions ?? []).filter { $0.modelID == modelID }
    }
}

extension LibraryStore {
    @discardableResult func registerANE(modelID: String, manifest: URL) throws -> LibraryANEPartition {
        var item = try LibraryANEPartition.inspect(modelID: modelID, manifest: manifest)
        let lease = try acquireLease(); defer { withExtendedLifetime(lease) {} }
        var index = try read()
        var items = index.anePartitions ?? []
        if let position = items.firstIndex(where: { $0.modelID == item.modelID && $0.path == item.path }) {
            item.id = items[position].id; items[position] = item
        } else { items.append(item) }
        index.anePartitions = items
        try write(index)
        return item
    }
    func unregisterANE(id: String) throws {
        let lease = try acquireLease(); defer { withExtendedLifetime(lease) {} }
        var index = try read()
        index.anePartitions = (index.anePartitions ?? []).filter { $0.id != id }
        try write(index)
    }
}
