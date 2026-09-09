import Foundation

@main struct InstallationInspectionTests {
    static func main() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-installation-\(UUID())")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw LibraryFailure(message: message) }
        }
        func write(_ path: String, _ data: Data) throws {
            let file = root.appendingPathComponent(path)
            try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: file)
        }
        func json(_ path: String, _ value: [String: Any]) throws {
            try write(path, JSONSerialization.data(withJSONObject: value))
        }
        func tensor(_ path: String, end: Any = 2, payload: Int = 2) throws {
            let header = try JSONSerialization.data(withJSONObject: ["weight": ["dtype": "BF16", "shape": [1], "data_offsets": [0, end]]])
            var size = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &size) { Data($0) }
            data.append(header); data.append(Data(repeating: 0, count: payload)); try write(path, data)
        }
        func inspect(_ model: String = "z-image-turbo", at path: String = "z") -> InstallationInspection {
            InstallationInspection.inspect(modelID: model, root: root.appendingPathComponent(path))
        }
        try check(inspect().issues.contains { $0.code == "missing_root" }, "Missing directory accepted")
        try json("z/tokenizer/tokenizer.json", ["model": ["type": "BPE"]])
        for component in ["transformer", "text_encoder", "vae"] { try tensor("z/\(component)/model.safetensors") }
        try check(inspect().status == "files_present" && inspect().checkedWeightFiles == 3, "Complete Z fixture rejected")
        try json("z/transformer/model.safetensors.index.json", ["weight_map": ["weight": "missing.safetensors"]])
        try check(inspect().issues.contains { $0.path.hasSuffix("missing.safetensors") }, "Missing indexed shard accepted")
        try json("z/transformer/model.safetensors.index.json", ["weight_map": ["weight": "../../outside.safetensors"]])
        try check(inspect().issues.contains { $0.code == "invalid_index" }, "Traversal in index accepted")
        try fm.removeItem(at: root.appendingPathComponent("z/transformer/model.safetensors.index.json"))
        try tensor("z/vae/model.safetensors", end: 1000)
        try check(inspect().issues.contains { $0.code == "invalid_weights" }, "Truncated tensor accepted")
        try tensor("z/vae/model.safetensors", end: true)
        try check(inspect().status == "incomplete", "Boolean byte offset accepted")
        try tensor("z/vae/model.safetensors")
        try fm.moveItem(at: root.appendingPathComponent("z/text_encoder"), to: root.appendingPathComponent("shared-text"))
        try fm.createSymbolicLink(atPath: root.appendingPathComponent("z/text_encoder").path, withDestinationPath: "../shared-text")
        try check(inspect().status == "files_present", "Shared component directory rejected")
        try fm.moveItem(at: root.appendingPathComponent("shared-text"), to: root.appendingPathComponent("moved-text"))
        try check(inspect().issues.contains { $0.path == "text_encoder" }, "Broken component link accepted")
        try fm.removeItem(at: root.appendingPathComponent("z/text_encoder"))
        try fm.createDirectory(at: root.appendingPathComponent("z/text_encoder"), withIntermediateDirectories: true)
        try fm.createSymbolicLink(atPath: root.appendingPathComponent("z/text_encoder/model.safetensors").path, withDestinationPath: "../../moved-text/model.safetensors")
        try check(inspect().status == "files_present", "Z-Image explicit single-file link rejected")
        try json("flux/tokenizer/tokenizer.json", ["model": ["type": "BPE"]])
        try json("flux/transformer/config.json", ["attention_head_dim": 128, "in_channels": 128, "num_attention_heads": 24,
                                                  "num_layers": 5, "num_single_layers": 20, "joint_attention_dim": 7680, "guidance_embeds": false])
        try json("flux/text_encoder/config.json", ["hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8])
        try json("flux/vae/config.json", ["latent_channels": 32])
        for component in ["transformer", "text_encoder", "vae"] { try tensor("flux/\(component)/model.safetensors") }
        try check(inspect("flux2-klein-4b", at: "flux").status == "files_present", "FLUX 4B configuration rejected")
        try check(inspect("flux2-klein-9b", at: "flux").issues.contains { $0.code == "incompatible_config" }, "4B installed as 9B accepted")
        for path in LibraryRecipe.all.first(where: { $0.modelID == "ltx-2.5-distilled" })!.include where path.hasSuffix(".safetensors") {
            try tensor("ltx/" + path)
        }
        try json("ltx/gemma4-12b-ltx-v1/tokenizer.json", ["model": ["type": "BPE"]])
        try check(inspect("ltx-2.5-distilled", at: "ltx").status == "files_present", "LTX required layout rejected")
        try json("h3/FL2VA/tokenizer/tokenizer.json", ["model": ["type": "BPE"]])
        try json("h3/FL2VA/transformer/config.json", ["model_type": "fixture"])
        try json("h3/FL2VA/transformer/model.safetensors.index.json", ["weight_map": ["weight": "model.safetensors"]])
        for component in ["text_encoder", "transformer", "video_vae/source", "audio_vae"] { try tensor("h3/FL2VA/\(component)/model.safetensors") }
        try check(inspect("minimax-h3-turbo", at: "h3").status == "needs_preparation", "Unmerged H3 falsely ready")
        try json("fast/mlx_dit.json", ["model_type": "fixture"]); try tensor("fast/mlx_dit.safetensors")
        try json("fast/tokenizer/tokenizer.json", ["model": ["type": "BPE"]])
        try tensor("fast/text_encoder/model.safetensors"); try tensor("fast/vae/model.safetensors")
        let fast = inspect("fastmetal-1.3b-qad", at: "fast")
        try check(fast.status == "needs_preparation", "FastMetal worker dependencies omitted")
        let roundtrip = try JSONDecoder().decode(InstallationInspection.self, from: JSONEncoder().encode(fast))
        try check(roundtrip.status == fast.status && roundtrip.checkedWeightFiles == 3, "CLI/App report roundtrip failed")
        print("PASS: six model families, missing shards, unsafe paths, truncation, invalid offsets, shared/broken links, FLUX variant mismatch, video preparation and report transport")
    }
}
