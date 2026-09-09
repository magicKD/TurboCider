import Foundation

struct LibrarySettings: Codable, Sendable {
    var modelRoot: String
    static var file: URL {
        if let path = ProcessInfo.processInfo.environment["TURBOCIDER_LIBRARY_SETTINGS"] { return URL(fileURLWithPath: path) }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TurboCider/library-settings.json")
    }
    static func configure(_ root: URL) throws {
        guard ProcessInfo.processInfo.environment["TURBOCIDER_MODEL_LIBRARY"] == nil else {
            throw LibraryFailure(message: "TURBOCIDER_MODEL_LIBRARY overrides saved settings; change that environment variable first.")
        }
        let store = try LibraryStore(root: root)
        _ = try store.read()
        let file = file
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let lease = try LibraryLease(file.appendingPathExtension("lock")); defer { withExtendedLifetime(lease) {} }
        try JSONEncoder().encode(LibrarySettings(modelRoot: store.root.path)).write(to: file, options: .atomic)
    }
}

struct SharedTextComponents {
    /// Check the actual Qwen3-4B structure and shard index before proposing reuse.
    /// This establishes architectural compatibility, not identical fine-tuning.
    static func inspect(_ root: URL) throws -> [String: LibraryComponent] {
        let encoder = root.appendingPathComponent("text_encoder")
        let tokenizer = root.appendingPathComponent("tokenizer")
        let data = try Data(contentsOf: encoder.appendingPathComponent("config.json"))
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              config["model_type"] as? String == "qwen3", config["hidden_size"] as? Int == 2560,
              config["num_hidden_layers"] as? Int == 36, config["vocab_size"] as? Int == 151936 else {
            throw LibraryFailure(message: "共享编码器须为 Qwen3-4B（2560 hidden、36 层、151936 vocab）。")
        }
        let tokenConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizer.appendingPathComponent("tokenizer_config.json"))) as? [String: Any]
        guard ["Qwen2Tokenizer", "Qwen2TokenizerFast"].contains(tokenConfig?["tokenizer_class"] as? String ?? ""),
              FileManager.default.fileExists(atPath: tokenizer.appendingPathComponent("tokenizer.json").path) else {
            throw LibraryFailure(message: "共享 tokenizer 不符合 Qwen3 文本输入要求。")
        }
        let index = encoder.appendingPathComponent("model.safetensors.index.json")
        let files: [String]
        if FileManager.default.fileExists(atPath: index.path) {
            guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any],
                  let map = manifest["weight_map"] as? [String: String], !map.isEmpty else {
                throw LibraryFailure(message: "文本编码器分片索引无效。")
            }
            files = Array(Set(map.values))
        } else { files = ["model.safetensors"] }
        for file in files {
            try LibraryStore.validateRelativePath(file)
            let path = encoder.appendingPathComponent(file)
            guard let size = try path.resourceValues(forKeys: [.fileSizeKey]).fileSize, size > 8 else {
                throw LibraryFailure(message: "文本编码器权重缺失或不完整：\(file)")
            }
        }
        return ["text_encoder": LibraryComponent(path: encoder.resolvingSymlinksInPath().path, compatibility: "qwen3-4b-z-image"),
                "tokenizer": LibraryComponent(path: tokenizer.resolvingSymlinksInPath().path, compatibility: "qwen3-4b-z-image")]
    }
}
