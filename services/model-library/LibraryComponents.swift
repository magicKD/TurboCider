import Foundation
import CoreFoundation
import Darwin

/// URL resource flags describe a final symlink itself on macOS. Runtime paths
/// intentionally contain component links, so inspect the target's file type.
private func componentFileInfo(_ url: URL) -> stat? {
    var value = stat()
    return fstatat(AT_FDCWD, url.path, &value, 0) == 0 ? value : nil
}

struct LibraryResolvedComponent: Codable, Sendable {
    var kind: String
    var title: String
    var path: String
    var format: String
    var precision: String
    var status: String
}

struct LibraryComponentInventory: Codable, Sendable {
    var modelID: String
    var path: String
    var supported: Bool
    var detail: String
    var components: [LibraryResolvedComponent]
}

/// The base supplies tokenizer/processor assets and components not replaced.
/// All paths identify existing local files; composition never downloads weights.
struct LibraryCompositionRequest: Codable, Sendable {
    var modelID: String
    var basePath: String
    var name: String? = nil
    var components: [String: String] = [:]

    init(modelID: String, basePath: String, name: String? = nil, components: [String: String] = [:]) {
        self.modelID = modelID; self.basePath = basePath; self.name = name; self.components = components
    }
    private enum CodingKeys: String, CodingKey { case modelID, basePath, name, components }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decode(String.self, forKey: .modelID); basePath = try c.decode(String.self, forKey: .basePath)
        name = try c.decodeIfPresent(String.self, forKey: .name)
        components = try c.decodeIfPresent([String: String].self, forKey: .components) ?? [:]
    }

    static func decode(_ data: Data) throws -> Self {
        guard data.count <= 1024 * 1024,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys).isSubset(of: ["modelID", "basePath", "name", "components"]) else {
            throw LibraryFailure(message: "组件配置必须为不超过 1 MiB 的对象，仅支持 modelID、basePath、name 和 components。")
        }
        return try JSONDecoder().decode(Self.self, from: data)
    }
}

/// An immutable runtime-visible root: replacing a component creates a new
/// installation, so an already-loaded model cannot silently retain old weights.
enum LibraryComponents {
    private static let supported = ["qwen-image-2.1", "z-image-turbo", "flux2-klein-4b", "flux2-klein-9b"]
    private static let kinds = ["encoder", "dit", "vae"]
    private static let titles = ["encoder": "文本编码器", "dit": "DiT", "vae": "VAE"]
    private static let qwen: [String: [String]] = [
        "dit": ["diffusion_models/qwen_image_2.1_bf16.safetensors", "diffusion_models/qwen-image-2.1-Q4_K_M.gguf"],
        "encoder": ["text_encoders/qwen3vl_8b_bf16.safetensors", "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"],
        "vae": ["vae/qwen_image_2.1_vae_bf16.safetensors"]
    ]
    private static var fm: FileManager { .default }
    private static func exists(_ url: URL) -> Bool { fm.fileExists(atPath: url.path) }
    private static func directory(_ url: URL) -> Bool { componentFileInfo(url).map { ($0.st_mode & S_IFMT) == S_IFDIR } ?? false }
    private static func canonical(_ path: String) throws -> URL {
        guard path.hasPrefix("/"), !path.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
            throw LibraryFailure(message: "组件路径必须为无控制字符的绝对本机路径。")
        }
        let value = URL(fileURLWithPath: path).standardizedFileURL
        guard exists(value) else { throw LibraryFailure(message: "组件不存在或链接已失效：\(path)") }
        // Keep a file's logical extension (HF snapshot links may target a
        // hash-named blob). LibraryStore.link still pins the resolved target.
        return directory(value) ? value.resolvingSymlinksInPath() : value
    }
    private static func relatives(_ model: String, _ kind: String) -> [String] {
        if model == "qwen-image-2.1" { return qwen[kind]! }
        if model == "z-image-turbo" {
            let split: [String]
            switch kind {
            case "dit": split = ZImageVariant.all.filter(\.runnable).map { "diffusion_models/" + $0.filename }
            case "encoder": split = ["text_encoders/qwen_3_4b.safetensors"]
            default: split = ["vae/ae.safetensors"]
            }
            let legacy = ["split_files/", "models/", ""].flatMap { prefix in split.map { prefix + $0 } }
            let standard = kind == "dit" ? "transformer" : kind == "encoder" ? "text_encoder" : "vae"
            // Match the runtime's shared-encoder priority for a Comfy root.
            return kind == "encoder" ? [standard] + legacy : legacy + [standard]
        }
        return [kind == "dit" ? "transformer" : kind == "encoder" ? "text_encoder" : "vae"]
    }
    private static func find(_ model: String, _ kind: String, in root: URL) -> URL? {
        relatives(model, kind).map { root.appendingPathComponent($0) }.first { candidate in
            if model == "qwen-image-2.1" { return componentFileInfo(candidate).map { ($0.st_mode & S_IFMT) == S_IFREG } ?? false }
            if model == "z-image-turbo", kind == "encoder", directory(candidate) {
                return ((try? fm.contentsOfDirectory(atPath: candidate.path)) ?? []).contains { $0.hasSuffix(".safetensors") }
            }
            return exists(candidate)
        }
    }
    private static func selected(_ path: String, model: String, kind: String) throws -> URL {
        let value = try canonical(path)
        guard directory(value) else { return value }
        if let nested = find(model, kind, in: value) { return try canonical(nested.path) }
        if model == "qwen-image-2.1" {
            for relative in qwen[kind]! {
                let leaf = value.appendingPathComponent(URL(fileURLWithPath: relative).lastPathComponent)
                if exists(leaf) { return try canonical(leaf.path) }
            }
            let files = try fm.contentsOfDirectory(at: value, includingPropertiesForKeys: nil)
                .filter { ["safetensors", "gguf"].contains($0.pathExtension) }
            guard files.count == 1 else { throw LibraryFailure(message: "请选择明确的 Qwen \(titles[kind]!) 权重文件，或仅含该组件的目录。") }
            return try canonical(files[0].path)
        }
        return value
    }
    static func inventory(modelID: String, root: URL) throws -> LibraryComponentInventory {
        let base = try canonical(root.path)
        guard directory(base) else { throw LibraryFailure(message: "请选择完整模型安装目录。") }
        let allowed = supported.contains(modelID)
        return .init(modelID: modelID, path: root.standardizedFileURL.path, supported: allowed,
            detail: allowed ? "替换会建立独立链接安装，保留原文件；文件检查不等于推理或画质验收。Qwen GGUF 仅限现有实验性近似文生图路线，不支持编辑或 LoRA。" :
                "此模型尚不支持独立 Encoder / DiT / VAE 替换；请导入完整安装。视频模型还需要专属投影、音频或上采样组件。",
            components: allowed ? kinds.map { kind in
                let value = find(modelID, kind, in: base)
                let ext = value?.pathExtension ?? ""
                let format = value.map { directory($0) ? "directory" : ext } ?? ""
                let name = value?.lastPathComponent.lowercased() ?? ""
                let precision = ext == "gguf" ? "Q4_K_M" : name.contains("bf16") ? "BF16" : name.contains("int8") ? "INT8 ConvRot" : name.contains("nvfp4") ? "NVFP4" : "按配置"
                return .init(kind: kind, title: titles[kind]!, path: value?.path ?? "", format: format,
                    precision: value == nil ? "未知" : precision, status: value == nil ? "missing" : "present")
            } : [])
    }

    static func compose(_ request: LibraryCompositionRequest, store: LibraryStore) throws -> LibraryInstallation {
        guard supported.contains(request.modelID) else {
            throw LibraryFailure(message: "\(request.modelID) 尚不支持独立三组件替换，请导入完整模型目录。")
        }
        guard Set(request.components.keys).isSubset(of: Set(kinds)) else {
            throw LibraryFailure(message: "components 仅支持 encoder、dit 和 vae。")
        }
        if let name = request.name, name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || name.utf8.count > 512 {
            throw LibraryFailure(message: "安装名称不能为空或超过 512 字节。")
        }
        let base = try canonical(request.basePath)
        guard directory(base) else { throw LibraryFailure(message: "basePath 必须为完整模型安装目录。") }
        var sources: [String: URL] = [:]
        for kind in kinds {
            try Task.checkCancellation()
            if let replacement = request.components[kind] {
                sources[kind] = try selected(replacement, model: request.modelID, kind: kind)
            } else if let existing = find(request.modelID, kind, in: base) {
                sources[kind] = try canonical(existing.path)
            } else { throw LibraryFailure(message: "原安装缺少 \(titles[kind]!)，请先选择该组件。") }
        }
        let lease = try store.acquireLease(); defer { withExtendedLifetime(lease) {} }
        let staging = try store.managedDirectory("staging").appendingPathComponent("components-" + UUID().uuidString)
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: staging) }
        var components: [String: LibraryComponent] = [:]
        func link(_ source: URL, _ destination: String, _ identity: String) throws {
            try Task.checkCancellation()
            try LibraryStore.link(source, at: destination, in: staging)
            components[destination] = .init(path: source.resolvingSymlinksInPath().path, compatibility: identity)
        }
        if request.modelID == "qwen-image-2.1" {
            for kind in kinds {
                let source = sources[kind]!
                try validateQwen(source, kind: kind)
                let destination = qwen[kind]!.first { URL(fileURLWithPath: $0).pathExtension == source.pathExtension }!
                try link(source, destination, "qwen-image-2.1/\(kind)/\(source.pathExtension == "gguf" ? "Q4_K_M" : "BF16")")
            }
            let processor = try canonical(base.appendingPathComponent("processor").path)
            guard directory(processor) else { throw LibraryFailure(message: "原安装缺少 processor 目录，无法继承 tokenizer。") }
            try link(processor, "processor", "qwen-image-2.1/processor")
        } else if request.modelID.hasPrefix("flux2-klein-") {
            for kind in kinds {
                guard directory(sources[kind]!) else { throw LibraryFailure(message: "FLUX \(titles[kind]!) 需选择含 config.json 和分片的组件目录，不能仅选择权重文件。") }
                try validateDirectory(sources[kind]!, model: request.modelID, kind: kind)
                try link(sources[kind]!, relatives(request.modelID, kind)[0], "\(request.modelID)/\(kind)")
            }
            try link(try canonical(base.appendingPathComponent("tokenizer").path), "tokenizer", "\(request.modelID)/tokenizer")
        } else {
            let split = !directory(sources["dit"]!)
            guard split == !directory(sources["vae"]!) else {
                throw LibraryFailure(message: "Z-Image 的 DiT 和 VAE 需使用同一布局：均为 Comfy 文件，或均为 Diffusers 组件目录。")
            }
            for kind in kinds {
                let source = sources[kind]!
                if directory(source) { try validateDirectory(source, model: request.modelID, kind: kind) }
                else { try validateZImageFile(source, kind: kind) }
                let destination: String
                if kind == "encoder" { destination = directory(source) ? "text_encoder" : "split_files/text_encoders/qwen_3_4b.safetensors" }
                else if !split { destination = kind == "dit" ? "transformer" : "vae" }
                else if kind == "dit" { destination = "split_files/diffusion_models/" + source.lastPathComponent }
                else { destination = "split_files/vae/ae.safetensors" }
                if !split && kind == "encoder" && !directory(source) {
                    // Z-Image explicitly supports a single linked shard inside a component directory.
                    try link(source, "text_encoder/model.safetensors", "z-image-turbo/encoder")
                } else { try link(source, destination, "z-image-turbo/\(kind)") }
            }
            try link(try canonical(base.appendingPathComponent("tokenizer").path), "tokenizer", "qwen3-4b-z-image/tokenizer")
        }
        let inspection = InstallationInspection.inspect(modelID: request.modelID, root: staging)
        guard inspection.issues.isEmpty else {
            throw LibraryFailure(message: "组件组合未通过安装检查：" + inspection.issues.map { "\($0.path)：\($0.message)" }.joined(separator: "；"))
        }
        try Task.checkCancellation()
        return try store.publish(staging: staging, modelID: request.modelID,
            name: request.name ?? "\(base.lastPathComponent) · 自定义组件", provider: "local-components",
            repository: base.path, revision: "linked-v1", components: components)
    }

    private static func requireShape(_ tensors: [String: ComponentTensor], _ names: [String], _ shape: [UInt64], _ description: String) throws {
        guard names.contains(where: { tensors[$0]?.shape == shape }) else {
            throw LibraryFailure(message: "\(description) 架构不匹配，缺少预期张量或 shape：\(names[0]) \(shape)。")
        }
    }
    static func validateQwen(_ file: URL, kind: String) throws {
        let gguf = file.pathExtension == "gguf"
        guard ["safetensors", "gguf"].contains(file.pathExtension), !directory(file), !(gguf && kind == "vae") else {
            throw LibraryFailure(message: "Qwen \(titles[kind]!) 需要支持的单文件 safetensors 或 Q4_K_M GGUF。")
        }
        if gguf, !qwen[kind]!.contains(where: { URL(fileURLWithPath: $0).lastPathComponent == file.lastPathComponent }) {
            throw LibraryFailure(message: "当前 Qwen GGUF 路线仅支持规范名称的 Q4_K_M 组件；Q5/Q6/Q8 等尚未接入，不能仅重命名。")
        }
        let tensors = try ComponentWeightHeader.read(file)
        let description = "Qwen \(titles[kind]!)"
        if !gguf, tensors.values.contains(where: { !["BF16", "F16", "F32", "I64"].contains($0.dtype) }) {
            throw LibraryFailure(message: "Qwen safetensors 组件只支持现有浮点权重路线；不能将量化权重重命名为 BF16。")
        }
        if kind == "dit" {
            let prefix = gguf ? "model.diffusion_model." : ""
            try requireShape(tensors, [prefix + "img_in.weight"], [4096, 64], description)
            try requireShape(tensors, [prefix + "txt_in.in_layer.weight"], [4096, 4096], description)
            try requireShape(tensors, [prefix + "transformer_blocks.31.attn.norm_k.weight"], [128], description)
            try requireShape(tensors, [prefix + "proj_out.weight"], [64, 4096], description)
        } else if kind == "encoder" {
            try requireShape(tensors, gguf ? ["token_embd.weight"] : ["model.embed_tokens.weight", "model.language_model.embed_tokens.weight"], [151936, 4096], description)
            try requireShape(tensors, gguf ? ["blk.35.attn_q.weight"] : ["model.layers.35.self_attn.q_proj.weight", "model.language_model.layers.35.self_attn.q_proj.weight"], [4096, 4096], description)
            if !gguf {
                try requireShape(tensors, ["model.visual.patch_embed.proj.weight", "visual.patch_embed.proj.weight"], [1152, 3, 2, 16, 16], description)
            }
        } else {
            let post = tensors["conv2.weight"] ?? tensors["post_quant_conv.weight"]
            let decoder = tensors["decoder.conv1.weight"] ?? tensors["decoder.conv_in.weight"]
            guard [[64, 64, 1, 1, 1], [64, 64, 1, 1]].contains(post?.shape ?? []),
                  [[1152, 64, 1, 3, 3], [1152, 64, 3, 3]].contains(decoder?.shape ?? []) else {
                throw LibraryFailure(message: "Qwen VAE 架构不匹配：需要 64-channel 单帧 VAE。")
            }
        }
    }
    private static func validateZImageFile(_ file: URL, kind: String) throws {
        guard file.pathExtension == "safetensors", !directory(file) else { throw LibraryFailure(message: "Z-Image 单文件组件必须为 safetensors。") }
        let tensors = try ComponentWeightHeader.read(file)
        if kind == "dit" {
            guard ZImageVariant.all.filter(\.runnable).contains(where: { $0.filename == file.lastPathComponent }) else {
                throw LibraryFailure(message: "请选择现有 Z-Image BF16 / INT8 ConvRot / NVFP4 规范文件，保持原生运行路线。")
            }
            try requireShape(tensors, ["cap_embedder.0.weight"], [2560], "Z-Image DiT")
            try requireShape(tensors, ["cap_embedder.1.bias"], [3840], "Z-Image DiT")
        } else if kind == "encoder" {
            try requireShape(tensors, ["model.embed_tokens.weight"], [151936, 2560], "Qwen3-4B 文本编码器")
            try requireShape(tensors, ["model.layers.35.input_layernorm.weight"], [2560], "Qwen3-4B 文本编码器")
        } else {
            try requireShape(tensors, ["decoder.conv_in.weight"], [512, 16, 3, 3], "Z-Image VAE")
            guard tensors["decoder.mid.block_1.conv1.weight"] != nil else {
                throw LibraryFailure(message: "Comfy Z-Image VAE 需要原生 ae.safetensors 命名；Diffusers VAE 请选择完整组件目录。")
            }
        }
    }
    private static func validateDirectory(_ source: URL, model: String, kind: String) throws {
        let configURL = source.appendingPathComponent("config.json")
        let config = try ComponentWeightHeader.json(configURL)
        let tensors = try ComponentWeightHeader.directory(source, allowSingleLink: model == "z-image-turbo")
        func integer(_ key: String) -> UInt64? { ComponentWeightHeader.integer(config[key]) }
        if model.hasPrefix("flux2-klein-") {
            let small = model == "flux2-klein-4b"
            if kind == "dit" {
                guard integer("attention_head_dim") == 128, integer("in_channels") == 128,
                      integer("num_attention_heads") == (small ? 24 : 32), integer("num_layers") == (small ? 5 : 8),
                      integer("num_single_layers") == (small ? 20 : 24), config["guidance_embeds"] as? Bool != true else {
                    throw LibraryFailure(message: "DiT 配置与所选 FLUX.2 Klein 型号不匹配。")
                }
                try requireShape(tensors, ["proj_out.weight"], [128, small ? 3072 : 4096], "FLUX DiT")
            } else if kind == "encoder" {
                guard integer("num_hidden_layers") == 36, integer("num_attention_heads") == 32,
                      integer("num_key_value_heads") == 8, let hidden = integer("hidden_size"), hidden > 0, hidden <= 100_000 else {
                    throw LibraryFailure(message: "FLUX 文本编码器配置不兼容。")
                }
                try requireShape(tensors, ["model.embed_tokens.weight"], [151936, hidden], "FLUX 文本编码器")
            } else {
                guard integer("latent_channels") == 32 else { throw LibraryFailure(message: "FLUX VAE 必须为 32 latent channels。") }
                try requireShape(tensors, ["decoder.conv_in.weight"], [512, 32, 3, 3], "FLUX VAE")
            }
        } else if kind == "encoder" {
            guard config["model_type"] as? String == "qwen3", integer("hidden_size") == 2560,
                  integer("num_hidden_layers") == 36, integer("vocab_size") == 151936 else {
                throw LibraryFailure(message: "Z-Image 编码器须为 Qwen3-4B（2560 hidden、36 层、151936 vocab）。")
            }
            if let quant = config["quantization"] as? [String: Any] {
                guard quant["mode"] as? String == "affine", ComponentWeightHeader.integer(quant["group_size"]) == 32,
                      [4, 8].contains(ComponentWeightHeader.integer(quant["bits"]) ?? 0), config["turbocider_dense_embedding"] as? Bool == true else {
                    throw LibraryFailure(message: "量化共享编码器需要现有 MLX affine Q4/Q8 group32 与 BF16 embedding 转换格式。")
                }
            }
            try requireShape(tensors, ["model.embed_tokens.weight"], [151936, 2560], "Qwen3-4B 文本编码器")
        } else if kind == "dit" {
            try requireShape(tensors, ["cap_embedder.0.weight", "transformer.cap_embedder.0.weight"], [2560], "Z-Image DiT")
            try requireShape(tensors, ["cap_embedder.1.bias", "transformer.cap_embedder.1.bias"], [3840], "Z-Image DiT")
        } else { try requireShape(tensors, ["decoder.conv_in.weight", "vae.decoder.conv_in.weight"], [512, 16, 3, 3], "Z-Image VAE") }
    }
}

struct ComponentTensor {
    var shape: [UInt64]
    var dtype: String
}

/// Header-only validation. Even a multi-gigabyte checkpoint uses bounded reads,
/// never mmap/Data(contentsOf:) of its tensor payload or MLX/Core ML allocation.
enum ComponentWeightHeader {
    static func integer(_ value: Any?) -> UInt64? {
        guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID() else { return nil }
        return UInt64(value.stringValue)
    }
    static func json(_ file: URL) throws -> [String: Any] {
        guard let info = componentFileInfo(file), (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size > 0, info.st_size <= 64 * 1024 * 1024,
              let object = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any], !object.isEmpty else {
            throw LibraryFailure(message: "组件 JSON 缺失、无效或超过 64 MiB：\(file.path)")
        }
        return object
    }
    static func directory(_ root: URL, allowSingleLink: Bool = false) throws -> [String: ComponentTensor] {
        let files = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isSymbolicLinkKey]).sorted { $0.path < $1.path }
        let shards = files.filter { $0.pathExtension == "safetensors" }
        guard !shards.isEmpty, shards.count <= 1024 else { throw LibraryFailure(message: "组件目录没有权重分片，或分片超过检查上限。") }
        var merged: [String: ComponentTensor] = [:]
        for file in shards {
            try Task.checkCancellation()
            let isLink = try file.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true
            guard (allowSingleLink && shards.count == 1) || !isLink else {
                throw LibraryFailure(message: "组件目录包含会被 MLX 跳过的单文件软链接，请选择原组件目录：\(file.path)")
            }
            for (key, tensor) in try read(file) {
                guard merged.updateValue(tensor, forKey: key) == nil else { throw LibraryFailure(message: "组件分片含重复张量：\(key)") }
            }
        }
        for index in files where index.lastPathComponent.hasSuffix(".safetensors.index.json") {
            guard let map = try json(index)["weight_map"] as? [String: String], !map.isEmpty else { throw LibraryFailure(message: "组件分片索引无效。") }
            for (key, shard) in map {
                try LibraryStore.validateRelativePath(shard)
                guard !shard.contains("/"), shards.contains(where: { $0.lastPathComponent == shard }), merged[key] != nil else {
                    throw LibraryFailure(message: "组件分片索引引用缺失或不安全的分片/张量：\(shard) / \(key)")
                }
            }
        }
        return merged
    }
    static func read(_ file: URL) throws -> [String: ComponentTensor] {
        if file.pathExtension == "gguf" { return try ComponentGGUFHeader.read(file) }
        guard file.pathExtension == "safetensors" else { throw LibraryFailure(message: "不支持的权重格式：\(file.pathExtension)") }
        guard let info = componentFileInfo(file), (info.st_mode & S_IFMT) == S_IFREG, info.st_size > 8 else { throw LibraryFailure(message: "权重缺失、非普通文件或为空。") }
        let size = info.st_size
        let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else { throw LibraryFailure(message: "safetensors 文件头不完整。") }
        let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        guard length > 0, length <= 64 * 1024 * 1024, length <= UInt64(size) - 8,
              let data = try handle.read(upToCount: Int(length)), data.count == Int(length),
              let header = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw LibraryFailure(message: "safetensors 文件头无效或被截断。") }
        let available = UInt64(size) - 8 - length
        var tensors: [String: ComponentTensor] = [:]
        var ranges: [(UInt64, UInt64)] = []
        let widths: [String: UInt64] = ["BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1]
        for (key, raw) in header where key != "__metadata__" {
            guard let tensor = raw as? [String: Any], let rawShape = tensor["shape"] as? [Any], rawShape.count <= 8,
                  let dtype = tensor["dtype"] as? String, let width = widths[dtype],
                  let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                  let start = integer(offsets[0]), let end = integer(offsets[1]), start <= end, end <= available else {
                throw LibraryFailure(message: "safetensors 张量描述或字节范围无效：\(key)")
            }
            var elements: UInt64 = 1; var shape: [UInt64] = []
            for raw in rawShape {
                guard let dim = integer(raw), dim <= UInt64(Int32.max) else { throw LibraryFailure(message: "无效张量维度：\(key)") }
                let product = elements.multipliedReportingOverflow(by: dim)
                guard !product.overflow else { throw LibraryFailure(message: "张量大小溢出：\(key)") }
                elements = product.partialValue; shape.append(dim)
            }
            let bytes = elements.multipliedReportingOverflow(by: width)
            guard !bytes.overflow, bytes.partialValue == end - start else { throw LibraryFailure(message: "张量 shape 与实际字节数不符：\(key)") }
            tensors[key] = .init(shape: shape, dtype: dtype); ranges.append((start, end))
        }
        guard !tensors.isEmpty else { throw LibraryFailure(message: "权重文件没有张量。") }
        ranges.sort { $0.0 < $1.0 }
        for i in 1..<max(1, ranges.count) where ranges[i].0 < ranges[i - 1].1 { throw LibraryFailure(message: "safetensors 张量字节范围重叠。") }
        return tensors
    }
}

/// GGUF directory reader mirrors the native format bounds and accepted tensor
/// types. Names/shapes establish the selected component's architecture only;
/// runtime remains responsible for actual route qualification and tensor math.
private final class ComponentGGUFHeader {
    let handle: FileHandle
    let size: UInt64
    var cursor: UInt64 = 0
    private var buffer = Data()
    private var bufferOffset = 0
    init(_ file: URL) throws {
        guard let info = componentFileInfo(file), (info.st_mode & S_IFMT) == S_IFREG, info.st_size > 24 else { throw LibraryFailure(message: "GGUF 权重缺失、非普通文件或为空。") }
        let bytes = info.st_size
        size = UInt64(bytes); handle = try FileHandle(forReadingFrom: file)
    }
    deinit { try? handle.close() }
    private func bytes(_ count: UInt64) throws -> Data {
        guard count <= 64 * 1024 * 1024, cursor <= min(size, 64 * 1024 * 1024), count <= min(size, 64 * 1024 * 1024) - cursor else {
            throw LibraryFailure(message: "GGUF 目录无效、被截断或超过 64 MiB。")
        }
        var data = Data(); data.reserveCapacity(Int(count))
        var remaining = Int(count)
        while remaining > 0 {
            if bufferOffset == buffer.count {
                buffer = try handle.read(upToCount: 64 * 1024) ?? Data(); bufferOffset = 0
                guard !buffer.isEmpty else { throw LibraryFailure(message: "GGUF 目录被截断。") }
            }
            let take = min(remaining, buffer.count - bufferOffset)
            data.append(buffer[bufferOffset..<(bufferOffset + take)])
            bufferOffset += take; remaining -= take
        }
        cursor += count; return data
    }
    private func number(_ width: UInt64) throws -> UInt64 { try bytes(width).enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) } }
    private func string() throws -> String {
        let length = try number(8)
        guard length <= 1024 * 1024, let value = String(data: try bytes(length), encoding: .utf8), !value.contains("\0") else { throw LibraryFailure(message: "GGUF 字符串无效或过长。") }
        return value
    }
    private func value(_ type: UInt64, depth: Int = 0) throws -> UInt64? {
        if type == 8 { _ = try string(); return nil }
        if type == 9 {
            guard depth < 4 else { throw LibraryFailure(message: "GGUF 元数据嵌套过深。") }
            let child = try number(4), count = try number(8)
            guard count <= 1024 * 1024 else { throw LibraryFailure(message: "GGUF 元数据数组过大。") }
            let widths: [UInt64: UInt64] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]
            if let width = widths[child] { _ = try bytes(width * count); return nil }
            guard child == 8 || child == 9 else { throw LibraryFailure(message: "GGUF 数组类型无效。") }
            for _ in 0..<count { _ = try value(child, depth: depth + 1) }
            return nil
        }
        let widths: [UInt64: UInt64] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]
        guard let width = widths[type] else { throw LibraryFailure(message: "GGUF 元数据类型无效。") }
        return try number(width)
    }
    static func read(_ file: URL) throws -> [String: ComponentTensor] {
        let reader = try Self(file)
        guard try reader.number(4) == 0x46554747, [2, 3].contains(try reader.number(4)) else { throw LibraryFailure(message: "GGUF magic 或版本无效。") }
        let count = try reader.number(8), metadata = try reader.number(8)
        guard count > 0, count <= 65536, metadata <= 65536 else { throw LibraryFailure(message: "GGUF 张量或元数据数量超限。") }
        var alignment: UInt64 = 32; var metadataKeys = Set<String>()
        for _ in 0..<metadata {
            let key = try reader.string(), type = try reader.number(4)
            guard !key.isEmpty, metadataKeys.insert(key).inserted else { throw LibraryFailure(message: "GGUF 元数据键为空或重复。") }
            let value = try reader.value(type)
            if key == "general.alignment" {
                guard type == 4, let value, value > 0, value <= 1 << 20, value & (value - 1) == 0 else { throw LibraryFailure(message: "GGUF alignment 无效。") }
                alignment = value
            }
        }
        // On-disk GGML block geometry, matching native/core/gguf_directory.hpp.
        let types: [UInt64: (UInt64, UInt64)] = [0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 30: (1, 2)]
        var tensors: [String: ComponentTensor] = [:]; var ranges: [(UInt64, UInt64)] = []
        for _ in 0..<count {
            try Task.checkCancellation()
            let name = try reader.string(), rank = try reader.number(4)
            guard !name.isEmpty, tensors[name] == nil, rank > 0, rank <= 4 else { throw LibraryFailure(message: "GGUF 张量名称/维度无效或重复。") }
            var shape: [UInt64] = []; var elements: UInt64 = 1
            for _ in 0..<rank {
                let dim = try reader.number(8), product = elements.multipliedReportingOverflow(by: dim)
                guard dim > 0, dim <= UInt64(Int32.max), !product.overflow else { throw LibraryFailure(message: "GGUF 张量大小无效。") }
                shape.append(dim); elements = product.partialValue
            }
            let type = try reader.number(4), offset = try reader.number(8)
            guard let (block, width) = types[type], shape[0] % block == 0, offset % alignment == 0 else { throw LibraryFailure(message: "不支持的 GGUF 张量类型或块对齐。") }
            let bytes = (elements / block).multipliedReportingOverflow(by: width)
            guard !bytes.overflow, offset <= reader.size, bytes.partialValue <= reader.size - offset else { throw LibraryFailure(message: "GGUF 张量字节范围超出文件。") }
            tensors[name] = .init(shape: Array(shape.reversed()), dtype: "GGUF:\(type)")
            ranges.append((offset, offset + bytes.partialValue))
        }
        let dataOffset = (reader.cursor + alignment - 1) & ~(alignment - 1)
        guard dataOffset <= reader.size else { throw LibraryFailure(message: "GGUF 缺少张量 payload。") }
        ranges.sort { $0.0 < $1.0 }
        for (i, range) in ranges.enumerated() {
            guard range.1 <= reader.size - dataOffset, i == 0 || range.0 >= ranges[i - 1].1 else { throw LibraryFailure(message: "GGUF payload 被截断或张量范围重叠。") }
        }
        return tensors
    }
}
