import Foundation
import CoreFoundation
import Darwin

struct InstallationIssue: Codable, Sendable {
    var code: String
    var path: String
    var message: String
}

/// A bounded, read-only file check. It never loads tensors or starts a worker.
/// Native execution still verifies tensor names, architecture and provenance.
struct InstallationInspection: Codable, Sendable {
    var modelID: String
    var path: String
    var checkedAt = Date()
    var scope = "Required files, shard references and safetensors byte ranges; no tensor math, full-file hashes, GPU or ANE validation."
    var issues: [InstallationIssue] = []
    var preparation: [String] = []
    var checkedWeightFiles = 0
    var referencedWeightBytes: UInt64 = 0
    var status: String { !issues.isEmpty ? "incomplete" : preparation.isEmpty ? "files_present" : "needs_preparation" }
    var title: String {
        switch status {
        case "incomplete": return "组件缺失或无效"
        case "needs_preparation": return "还需准备运行环境"
        default: return "所需文件检查通过"
        }
    }
    // Include derived status for CLI consumers without trusting persisted results.
    enum CodingKeys: String, CodingKey { case modelID, path, checkedAt, scope, issues, preparation, checkedWeightFiles, referencedWeightBytes, status }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelID, forKey: .modelID); try c.encode(path, forKey: .path)
        try c.encode(checkedAt, forKey: .checkedAt); try c.encode(scope, forKey: .scope)
        try c.encode(issues, forKey: .issues); try c.encode(preparation, forKey: .preparation)
        try c.encode(checkedWeightFiles, forKey: .checkedWeightFiles)
        try c.encode(referencedWeightBytes, forKey: .referencedWeightBytes); try c.encode(status, forKey: .status)
    }
    init(modelID: String, path: String) { self.modelID = modelID; self.path = path }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decode(String.self, forKey: .modelID); path = try c.decode(String.self, forKey: .path)
        checkedAt = try c.decode(Date.self, forKey: .checkedAt); scope = try c.decode(String.self, forKey: .scope)
        issues = try c.decode([InstallationIssue].self, forKey: .issues)
        preparation = try c.decode([String].self, forKey: .preparation)
        checkedWeightFiles = try c.decode(Int.self, forKey: .checkedWeightFiles)
        referencedWeightBytes = try c.decode(UInt64.self, forKey: .referencedWeightBytes)
    }
    static func inspect(modelID: String, root: URL) -> Self {
        var probe = InstallationProbe(report: .init(modelID: modelID, path: root.standardizedFileURL.path))
        probe.run()
        return probe.report
    }
}

private struct InstallationProbe {
    var report: InstallationInspection
    private var root: URL { URL(fileURLWithPath: report.path) }
    private var seen = Set<String>()
    init(report: InstallationInspection) { self.report = report }
    private func url(_ relative: String) -> URL { root.appendingPathComponent(relative) }
    private func info(_ relative: String) -> stat? {
        var value = stat()
        return fstatat(AT_FDCWD, url(relative).path, &value, 0) == 0 ? value : nil
    }
    private func exists(_ relative: String) -> Bool { info(relative) != nil }
    private mutating func issue(_ code: String, _ path: String, _ message: String) {
        report.issues.append(.init(code: code, path: path, message: message))
    }
    private func readJSON(_ relative: String) throws -> [String: Any] {
        guard let value = info(relative), (value.st_mode & S_IFMT) == S_IFREG, value.st_size > 0, value.st_size <= 64 * 1024 * 1024,
              let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url(relative))) as? [String: Any], !object.isEmpty else {
            throw LibraryFailure(message: "JSON 缺失、无效或超过 64 MiB 检查上限。")
        }
        return object
    }
    @discardableResult private mutating func json(_ relative: String) -> [String: Any]? {
        do { return try readJSON(relative) }
        catch { issue("invalid_json", relative, error.localizedDescription); return nil }
    }
    private func integer(_ raw: Any?) -> UInt64? {
        guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
              let result = UInt64(n.stringValue) else { return nil }
        return result
    }
    private mutating func weights(_ relative: String) {
        guard seen.insert(relative).inserted else { return }
        do {
            guard let value = info(relative), (value.st_mode & S_IFMT) == S_IFREG, value.st_size > 8 else {
                throw LibraryFailure(message: "权重文件缺失、链接失效或为空。")
            }
            let file = try FileHandle(forReadingFrom: url(relative)); defer { try? file.close() }
            guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else { throw LibraryFailure(message: "权重文件头不完整。") }
            let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
            guard length > 0, length <= 64 * 1024 * 1024, length <= UInt64(value.st_size) - 8,
                  let data = try file.read(upToCount: Int(length)), data.count == Int(length),
                  let header = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw LibraryFailure(message: "safetensors 文件头无效、文件被截断或文件头超过 64 MiB。")
            }
            let tensors = header.filter { $0.key != "__metadata__" }
            guard !tensors.isEmpty else { throw LibraryFailure(message: "safetensors 不含张量。") }
            let available = UInt64(value.st_size) - 8 - length
            for (name, raw) in tensors {
                guard let tensor = raw as? [String: Any], let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                      let start = integer(offsets[0]), let end = integer(offsets[1]), start <= end, end <= available,
                      tensor["dtype"] is String, tensor["shape"] is [Any] else {
                    throw LibraryFailure(message: "张量字节范围或文件头无效（可能下载不完整）：\(name)")
                }
            }
            report.checkedWeightFiles += 1
            report.referencedWeightBytes += UInt64(value.st_size)
        } catch { issue("invalid_weights", relative, error.localizedDescription) }
    }
    private mutating func component(_ relative: String, mlxDirectory: Bool = false, allowSingleLink: Bool = false) {
        do {
            guard let value = info(relative), (value.st_mode & S_IFMT) == S_IFDIR else {
                throw LibraryFailure(message: "缺少组件目录或目录链接失效。")
            }
            let files = try FileManager.default.contentsOfDirectory(atPath: url(relative).path).sorted()
            let shards = files.filter { $0.hasSuffix(".safetensors") }
            let links = Set(shards.filter { (try? FileManager.default.destinationOfSymbolicLink(atPath: url(relative + "/" + $0).path)) != nil })
            // Z-Image explicitly supports one linked single-file component when
            // there are no regular shards. FLUX uses the generic MLX loader.
            let singleLinkedComponent = allowSingleLink && shards.count == 1 && links.count == 1
            let indexes = files.filter { $0.hasSuffix(".safetensors.index.json") }
            for index in indexes {
                guard let manifest = json(relative + "/" + index) else { continue }
                guard let map = manifest["weight_map"] as? [String: String], !map.isEmpty else {
                    issue("invalid_index", relative + "/" + index, "分片索引缺少 weight_map。"); continue
                }
                for shard in Set(map.values).sorted() {
                    do {
                        try LibraryStore.validateRelativePath(shard)
                        guard !shard.contains("/"), shard.hasSuffix(".safetensors") else { throw LibraryFailure(message: "运行时要求组件目录中的 safetensors 分片。") }
                        if mlxDirectory && !singleLinkedComponent && links.contains(shard) {
                            issue("unsupported_shard_link", relative + "/" + shard, "MLX 目录加载会跳过此分片软链接；请共享整个组件目录或使用硬链接。")
                        }
                        weights(relative + "/" + shard)
                    } catch { issue("invalid_index", relative + "/" + index, "不支持的分片路径：\(shard)") }
                }
            }
            var usable = 0
            for shard in shards {
                if mlxDirectory && !singleLinkedComponent && links.contains(shard) { continue }
                usable += 1; weights(relative + "/" + shard)
            }
            if usable == 0 { issue("missing_weights", relative, "未找到运行时可加载的权重分片；可共享整个组件目录，MLX 目录加载不会读取单个文件的软链接。") }
        } catch { issue("missing_component", relative, error.localizedDescription) }
    }
    mutating func run() {
        guard let value = info(""), (value.st_mode & S_IFMT) == S_IFDIR else {
            issue("missing_root", report.path, "模型目录不可访问或已移动；请选择现有目录重新登记。"); return
        }
        switch report.modelID {
        case "z-image-turbo":
            json("tokenizer/tokenizer.json")
            let comfy = "split_files/diffusion_models/z_image_turbo_bf16.safetensors"
            if exists(comfy) && exists("split_files/vae/ae.safetensors") {
                weights(comfy); weights("split_files/vae/ae.safetensors")
                if exists("split_files/text_encoders/qwen_3_4b.safetensors") { weights("split_files/text_encoders/qwen_3_4b.safetensors") }
                else { component("text_encoder", mlxDirectory: true, allowSingleLink: true) }
            } else {
                for path in ["transformer", "text_encoder", "vae"] { component(path, mlxDirectory: true, allowSingleLink: true) }
            }
        case "flux2-klein-4b", "flux2-klein-9b":
            json("tokenizer/tokenizer.json")
            let t = json("transformer/config.json"), q = json("text_encoder/config.json"), v = json("vae/config.json")
            let small = report.modelID == "flux2-klein-4b"
            if let t, (integer(t["attention_head_dim"]) != 128 || integer(t["in_channels"]) != 128 ||
                       integer(t["num_attention_heads"]) != (small ? 24 : 32) || integer(t["num_layers"]) != (small ? 5 : 8) ||
                       integer(t["num_single_layers"]) != (small ? 20 : 24) || t["guidance_embeds"] as? Bool == true) {
                issue("incompatible_config", "transformer/config.json", "Transformer 配置与所选 FLUX.2 Klein 型号不匹配。")
            }
            if let q, let t {
                let hidden = integer(q["hidden_size"]) ?? 0
                if hidden == 0 || hidden > 100_000 || integer(q["num_hidden_layers"]) != 36 || integer(q["num_attention_heads"]) != 32 ||
                    integer(q["num_key_value_heads"]) != 8 || integer(t["joint_attention_dim"]) != hidden * 3 {
                    issue("incompatible_config", "text_encoder/config.json", "文本编码器配置与所选 FLUX Transformer 不匹配。")
                }
            }
            if let v, integer(v["latent_channels"]) != 32 { issue("incompatible_config", "vae/config.json", "FLUX VAE 必须使用 32 个 latent channels。") }
            for path in ["transformer", "text_encoder", "vae"] { component(path, mlxDirectory: true) }
        case "ltx-2.5-distilled":
            for path in LibraryRecipe.all.first(where: { $0.modelID == report.modelID })!.include where path.hasSuffix(".safetensors") { weights(path) }
            json("gemma4-12b-ltx-v1/tokenizer.json")
        case "minimax-h3-turbo":
            json("FL2VA/tokenizer/tokenizer.json"); json("FL2VA/transformer/config.json")
            json("FL2VA/transformer/model.safetensors.index.json")
            for path in ["text_encoder", "transformer", "video_vae/source", "audio_vae"] { component("FL2VA/" + path) }
            let manifest = "FL2VA/transformer/h3-turbo-merge-manifest.json"
            if exists(manifest) {
                if let value = json(manifest), value["schema"] as? String != "h3-turbo-merge-manifest-v2" {
                    issue("invalid_preparation", manifest, "需要 prepare-lora 生成的 v2 来源记录。")
                }
            } else { report.preparation.append("H3 Turbo 需要 LightX2V Turbo LoRA：运行离线 prepare-lora 工具生成预融合模型。参见 docs/USAGE.md。") }
        case "wan2.1-1.3b-qad":
            json("mlx_dit.json"); weights("mlx_dit.safetensors"); json("tokenizer/tokenizer.json")
            component("text_encoder"); weights("vae/taew2_1.safetensors")
            report.preparation.append("Wan 需要原生 MLX/TAEHV 组件及模型目录中的 vae/taew2_1.safetensors；文件检查不运行 Python。")
        default: issue("unsupported_model", report.modelID, "此型号没有安装检查规则。")
        }
    }
}
