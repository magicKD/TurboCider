import Foundation
import Darwin

/// Header-only sparse fixtures: no real checkpoint, tensor math or model catalog.
@main struct LibraryComponentTests {
    static func main() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-component-tests-" + UUID().uuidString).standardizedFileURL.resolvingSymlinksInPath()
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let store = try LibraryStore(root: root.appendingPathComponent("library"))
        var assertions = 0
        func check(_ value: Bool, _ message: String) throws {
            assertions += 1
            if !value { throw LibraryFailure(message: message) }
        }
        func reject(_ message: String, _ body: () throws -> Void) throws {
            var rejected = false
            do { try body() } catch { rejected = true }
            try check(rejected, message)
        }
        func url(_ path: String) -> URL { root.appendingPathComponent(path) }
        func write(_ path: String, _ data: Data) throws {
            let target = url(path)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target)
        }
        func json(_ path: String, _ object: [String: Any]) throws {
            try write(path, JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))
        }
        func tensorFile(_ path: String, _ shapes: [String: [UInt64]], dtype: String = "BF16") throws {
            var header: [String: Any] = [:]; var offset: UInt64 = 0
            for key in shapes.keys.sorted() {
                let shape = shapes[key]!, size = shape.reduce(UInt64(2), *)
                header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]]
                offset += size
            }
            let metadata = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
            var length = UInt64(metadata.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(metadata)
            try write(path, data)
            let handle = try FileHandle(forWritingTo: url(path)); defer { try? handle.close() }
            // ftruncate creates holes. Only the small header is written/read;
            // the virtual size makes shape/byte-range validation meaningful.
            try handle.truncate(atOffset: UInt64(data.count) + offset)
        }
        let qwenDit: [String: [UInt64]] = [
            "img_in.weight": [4096, 64], "txt_in.in_layer.weight": [4096, 4096],
            "transformer_blocks.31.attn.norm_k.weight": [128], "proj_out.weight": [64, 4096]
        ]
        let qwenEncoder: [String: [UInt64]] = [
            "model.embed_tokens.weight": [151936, 4096],
            "model.layers.35.self_attn.q_proj.weight": [4096, 4096],
            "model.visual.patch_embed.proj.weight": [1152, 3, 2, 16, 16]
        ]
        let qwenVAE: [String: [UInt64]] = ["conv2.weight": [64, 64, 1, 1, 1], "decoder.conv1.weight": [1152, 64, 1, 3, 3]]
        try json("qwen/processor/tokenizer.json", ["model": ["type": "BPE"]])
        try tensorFile("qwen/diffusion_models/qwen_image_2.1_bf16.safetensors", qwenDit)
        try tensorFile("qwen/text_encoders/qwen3vl_8b_bf16.safetensors", qwenEncoder)
        try tensorFile("qwen/vae/qwen_image_2.1_vae_bf16.safetensors", qwenVAE)
        try tensorFile("alternate-encoder.safetensors", qwenEncoder)
        let original = try store.register(modelID: "qwen-image-2.1", path: url("qwen"))
        let inventory = try LibraryComponents.inventory(modelID: original.modelID, root: url("qwen"))
        try check(inventory.supported && inventory.components.map(\.kind) == ["encoder", "dit", "vae"], "Qwen component ordering changed")
        try check(inventory.components.allSatisfy { $0.status == "present" && $0.format == "safetensors" }, "Qwen components not discovered")
        try fm.createSymbolicLink(at: url("qwen-alias"), withDestinationURL: url("qwen"))
        let aliasedInventory = try LibraryComponents.inventory(modelID: original.modelID, root: url("qwen-alias"))
        try check(aliasedInventory.path == url("qwen-alias").standardizedFileURL.path, "Inventory did not echo the logical requested model root")
        let unsupported = try LibraryComponents.inventory(modelID: "ltx-2.5-distilled", root: url("qwen"))
        try check(!unsupported.supported && unsupported.components.isEmpty, "Complex video model falsely supports three-component replacement")
        let request = LibraryCompositionRequest(modelID: original.modelID, basePath: original.path,
            name: "Custom encoder", components: ["encoder": url("alternate-encoder.safetensors").path])
        func prefix(_ file: URL) throws -> Data {
            let handle = try FileHandle(forReadingFrom: file); defer { try? handle.close() }
            return try handle.read(upToCount: 4096) ?? Data()
        }
        let sourcePrefix = try prefix(url("alternate-encoder.safetensors"))
        let originalPrefix = try prefix(url("qwen/diffusion_models/qwen_image_2.1_bf16.safetensors"))
        let installation = try LibraryComponents.compose(request, store: store)
        let bound = URL(fileURLWithPath: installation.path)
        try check(installation.managed && installation.provider == "local-components" && installation.path != original.path,
                  "Composition did not publish a distinct managed runtime root")
        // URL equality also includes directory/trailing-slash metadata. These
        // assertions concern the exact filesystem target, not the URL flags.
        try check(bound.appendingPathComponent("text_encoders/qwen3vl_8b_bf16.safetensors").resolvingSymlinksInPath().standardizedFileURL.path == url("alternate-encoder.safetensors").standardizedFileURL.path, "Selected encoder did not affect runtime path")
        try check(bound.appendingPathComponent("diffusion_models/qwen_image_2.1_bf16.safetensors").resolvingSymlinksInPath().standardizedFileURL.path == url("qwen/diffusion_models/qwen_image_2.1_bf16.safetensors").standardizedFileURL.path, "Omitted DiT was not inherited")
        try check(bound.appendingPathComponent("processor").resolvingSymlinksInPath().standardizedFileURL.path == url("qwen/processor").standardizedFileURL.path, "Processor/tokenizer was not inherited")
        try check(try Data(contentsOf: bound.appendingPathComponent("processor/tokenizer.json")) == Data(contentsOf: url("qwen/processor/tokenizer.json")), "Inherited tokenizer bytes differ or are missing")
        try check(InstallationInspection.inspect(modelID: original.modelID, root: bound).status == "files_present", "Published Qwen binding failed installation inspection")
        let boundInventory = try LibraryComponents.inventory(modelID: original.modelID, root: bound)
        try check(boundInventory.components.allSatisfy { $0.status == "present" && $0.format == "safetensors" }, "Qwen component file links were not discovered by target type")
        try check((try store.read()).installations.count == 2, "Composition replaced the original registration")
        try check((try fm.destinationOfSymbolicLink(atPath: bound.appendingPathComponent("text_encoders/qwen3vl_8b_bf16.safetensors").path)) == url("alternate-encoder.safetensors").path, "Component bytes were copied instead of linked")
        try check((try fm.attributesOfItem(atPath: url("alternate-encoder.safetensors").path)[.referenceCount] as? NSNumber)?.intValue == 1, "Composition hard-linked or modified the original source")
        try check(fm.fileExists(atPath: original.path), "Source installation was removed")
        try check(try prefix(url("alternate-encoder.safetensors")) == sourcePrefix, "Selected source bytes changed")
        try check(try prefix(url("qwen/diffusion_models/qwen_image_2.1_bf16.safetensors")) == originalPrefix, "Original installation bytes changed")

        // Hash-named source targets must retain the selected file's logical
        // extension, like Hugging Face snapshot symlinks.
        try fm.moveItem(at: url("alternate-encoder.safetensors"), to: url("encoder-blob"))
        try fm.createSymbolicLink(at: url("alternate-encoder.safetensors"), withDestinationURL: url("encoder-blob"))
        let snapshot = try LibraryComponents.compose(request, store: store)
        try check(URL(fileURLWithPath: snapshot.path).appendingPathComponent("text_encoders/qwen3vl_8b_bf16.safetensors").resolvingSymlinksInPath().standardizedFileURL.path == url("encoder-blob").standardizedFileURL.path, "Hash-named linked source lost its logical safetensors type")
        try check((try fm.destinationOfSymbolicLink(atPath: URL(fileURLWithPath: snapshot.path).appendingPathComponent("text_encoders/qwen3vl_8b_bf16.safetensors").path)) == url("encoder-blob").path, "Published binding did not pin the resolved target")
        let chained = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: snapshot.path), store: store)
        let chainedRoot = URL(fileURLWithPath: chained.path)
        try check(chainedRoot.appendingPathComponent("text_encoders/qwen3vl_8b_bf16.safetensors").resolvingSymlinksInPath().standardizedFileURL.path == url("encoder-blob").standardizedFileURL.path, "Recomposition did not bind the ultimate encoder source")
        try check((try fm.destinationOfSymbolicLink(atPath: chainedRoot.appendingPathComponent("processor").path)) == url("qwen/processor").path, "Recomposition retained an intermediary tokenizer binding")
        let movedBinding = url("temporarily-moved-binding")
        try fm.moveItem(at: URL(fileURLWithPath: snapshot.path), to: movedBinding)
        try check(InstallationInspection.inspect(modelID: original.modelID, root: chainedRoot).status == "files_present", "New binding depended on the previous binding directory")
        try check(try Data(contentsOf: chainedRoot.appendingPathComponent("processor/tokenizer.json")) == Data(contentsOf: url("qwen/processor/tokenizer.json")), "Tokenizer depended on the moved intermediate binding")
        try fm.moveItem(at: movedBinding, to: URL(fileURLWithPath: snapshot.path))

        // Failure must not register or leave a partially published installation.
        let beforeFailure = try store.read().installations.count
        try reject("Encoder accepted as DiT") {
            _ = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path,
                components: ["dit": url("alternate-encoder.safetensors").path]), store: store)
        }
        try reject("Unknown component role accepted") {
            _ = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path,
                components: ["tokenizer": url("qwen/processor").path]), store: store)
        }
        try reject("Relative source accepted") {
            _ = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path, components: ["vae": "relative.safetensors"]), store: store)
        }
        try reject("Missing source accepted") {
            _ = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path, components: ["vae": url("missing.safetensors").path]), store: store)
        }
        try reject("Unsupported model composition accepted") {
            _ = try LibraryComponents.compose(.init(modelID: "wan2.1-1.3b-qad", basePath: original.path), store: store)
        }
        try reject("Unknown composition envelope key accepted") {
            _ = try LibraryCompositionRequest.decode(Data("{\"modelID\":\"qwen-image-2.1\",\"basePath\":\"/tmp\",\"unsafe\":true}".utf8))
        }
        try reject("Non-string component source accepted") {
            _ = try LibraryCompositionRequest.decode(Data("{\"modelID\":\"qwen-image-2.1\",\"basePath\":\"/tmp\",\"components\":{\"dit\":true}}".utf8))
        }
        let defaults = try LibraryCompositionRequest.decode(Data("{\"modelID\":\"qwen-image-2.1\",\"basePath\":\"/tmp\"}".utf8))
        try check(defaults.components.isEmpty, "Sparse composition defaults require unnecessary component fields")
        try check((try store.read()).installations.count == beforeFailure, "Rejected composition changed registry")
        try check((try fm.contentsOfDirectory(atPath: store.root.appendingPathComponent("staging").path)).isEmpty, "Rejected composition leaked private staging")

        func gguf(_ path: String, _ shapes: [String: [UInt64]]) throws {
            var data = Data()
            func integer(_ value: UInt64, _ bytes: Int) {
                for i in 0..<bytes { data.append(UInt8(truncatingIfNeeded: value >> (i * 8))) }
            }
            func string(_ value: String) { let bytes = Data(value.utf8); integer(UInt64(bytes.count), 8); data.append(bytes) }
            integer(0x46554747, 4); integer(3, 4); integer(UInt64(shapes.count), 8); integer(2, 8)
            string("general.alignment"); integer(4, 4); integer(32, 4)
            string("test.strings"); integer(9, 4); integer(8, 4); integer(2, 8); string("fixture"); string("猫")
            var offset: UInt64 = 0
            for key in shapes.keys.sorted() {
                let shape = shapes[key]!
                string(key); integer(UInt64(shape.count), 4)
                for dim in shape.reversed() { integer(dim, 8) }
                integer(1, 4); integer(offset, 8) // F16 tensors inside a GGUF container.
                offset += shape.reduce(UInt64(2), *)
                offset = (offset + 31) & ~UInt64(31)
            }
            let dataOffset = (UInt64(data.count) + 31) & ~UInt64(31)
            try write(path, data)
            let file = try FileHandle(forWritingTo: url(path)); defer { try? file.close() }
            try file.truncate(atOffset: dataOffset + offset)
        }
        let ggufPath = "qwen-image-2.1-Q4_K_M.gguf"
        try gguf(ggufPath, Dictionary(uniqueKeysWithValues: qwenDit.map { ("model.diffusion_model." + $0.key, $0.value) }))
        let mixed = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path,
            components: ["dit": url(ggufPath).path]), store: store)
        let mixedRoot = URL(fileURLWithPath: mixed.path)
        try check(!fm.fileExists(atPath: mixedRoot.appendingPathComponent("diffusion_models/qwen_image_2.1_bf16.safetensors").path), "Old BF16 would shadow the selected GGUF")
        try check(mixedRoot.appendingPathComponent("diffusion_models/" + ggufPath).resolvingSymlinksInPath().standardizedFileURL.path == url(ggufPath).standardizedFileURL.path, "GGUF replacement not visible to native component_source")
        let mixedReport = InstallationInspection.inspect(modelID: original.modelID, root: mixedRoot)
        try check(mixedReport.status == "needs_preparation" && mixedReport.checkedWeightFiles == 3 && mixedReport.preparation.contains { $0.contains("不支持图片编辑") }, "GGUF falsely advertised as full BF16 or omitted its runtime limits")
        try gguf("qwen-image-2.1-Q5_K_M.gguf", Dictionary(uniqueKeysWithValues: qwenDit.map { ("model.diffusion_model." + $0.key, $0.value) }))
        try reject("Unqualified Qwen GGUF precision accepted") {
            _ = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path,
                components: ["dit": url("qwen-image-2.1-Q5_K_M.gguf").path]), store: store)
        }
        try gguf("bad/qwen-image-2.1-Q4_K_M.gguf", ["unrelated.weight": [4, 4]])
        try reject("Unrelated GGUF accepted under a Qwen filename") {
            _ = try LibraryComponents.compose(.init(modelID: original.modelID, basePath: original.path,
                components: ["dit": url("bad/qwen-image-2.1-Q4_K_M.gguf").path]), store: store)
        }
        try write("bad/truncated.safetensors", Data(repeating: 255, count: 12))
        try reject("Invalid bounded safetensors header accepted") { _ = try ComponentWeightHeader.read(url("bad/truncated.safetensors")) }
        try write("bad/truncated.gguf", Data([0x47, 0x47, 0x55, 0x46]))
        try reject("Truncated GGUF accepted") { _ = try ComponentWeightHeader.read(url("bad/truncated.gguf")) }

        // FLUX directory replacement and cross-model shape/config safeguards.
        try json("flux/tokenizer/tokenizer.json", ["model": ["type": "BPE"]])
        try json("flux/transformer/config.json", ["attention_head_dim": 128, "in_channels": 128, "num_attention_heads": 24,
            "num_layers": 5, "num_single_layers": 20, "joint_attention_dim": 7680, "guidance_embeds": false])
        try json("flux/text_encoder/config.json", ["model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 151936])
        try json("flux/vae/config.json", ["latent_channels": 32])
        try tensorFile("flux/transformer/model.safetensors", ["proj_out.weight": [128, 3072]])
        try tensorFile("flux/text_encoder/model.safetensors", ["model.embed_tokens.weight": [151936, 2560]])
        try tensorFile("flux/vae/model.safetensors", ["decoder.conv_in.weight": [512, 32, 3, 3]])
        let flux = try LibraryComponents.compose(.init(modelID: "flux2-klein-4b", basePath: url("flux").path), store: store)
        try check(InstallationInspection.inspect(modelID: "flux2-klein-4b", root: URL(fileURLWithPath: flux.path)).status == "files_present", "Linked FLUX directory failed inspection")
        try check(URL(fileURLWithPath: flux.path).appendingPathComponent("transformer").resolvingSymlinksInPath().standardizedFileURL.path == url("flux/transformer").standardizedFileURL.path, "FLUX did not link the entire component directory")
        try reject("FLUX4B DiT accepted as 9B") {
            _ = try LibraryComponents.compose(.init(modelID: "flux2-klein-9b", basePath: url("flux").path), store: store)
        }
        try reject("FLUX accepts standalone weights without config") {
            _ = try LibraryComponents.compose(.init(modelID: "flux2-klein-4b", basePath: url("flux").path,
                components: ["dit": url("flux/transformer/model.safetensors").path]), store: store)
        }
        try fm.moveItem(at: url("flux/transformer/model.safetensors"), to: url("flux-dit.safetensors"))
        try fm.createSymbolicLink(at: url("flux/transformer/model.safetensors"), withDestinationURL: url("flux-dit.safetensors"))
        try reject("MLX-skipped FLUX shard link accepted") {
            _ = try LibraryComponents.compose(.init(modelID: "flux2-klein-4b", basePath: url("flux").path), store: store)
        }

        // Z-Image keeps Comfy paths and can share a real Qwen3-4B directory.
        try json("z/tokenizer/tokenizer.json", ["model": ["type": "BPE"]])
        try tensorFile("z/models/diffusion_models/z_image_turbo_bf16.safetensors",
            ["cap_embedder.0.weight": [2560], "cap_embedder.1.bias": [3840]])
        try tensorFile("z/models/vae/ae.safetensors", ["decoder.conv_in.weight": [512, 16, 3, 3], "decoder.mid.block_1.conv1.weight": [512, 512, 3, 3]])
        try tensorFile("z/models/text_encoders/qwen_3_4b.safetensors", ["model.embed_tokens.weight": [151936, 2560], "model.layers.35.input_layernorm.weight": [2560]])
        let z = try LibraryComponents.compose(.init(modelID: "z-image-turbo", basePath: url("z").path,
            components: ["encoder": url("flux/text_encoder").path]), store: store)
        let zRoot = URL(fileURLWithPath: z.path)
        try check(zRoot.appendingPathComponent("split_files/diffusion_models/z_image_turbo_bf16.safetensors").resolvingSymlinksInPath().standardizedFileURL.path == url("z/models/diffusion_models/z_image_turbo_bf16.safetensors").standardizedFileURL.path, "Z-Image legacy models/ layout was not normalized to actual runtime split_files/")
        try check(zRoot.appendingPathComponent("text_encoder").resolvingSymlinksInPath().standardizedFileURL.path == url("flux/text_encoder").standardizedFileURL.path, "Z-Image shared encoder not visible to runtime")
        let zInventory = try LibraryComponents.inventory(modelID: "z-image-turbo", root: zRoot)
        try check(zInventory.components.first { $0.kind == "encoder" }?.format == "directory", "Linked encoder directory returned an empty format")
        try check(InstallationInspection.inspect(modelID: "z-image-turbo", root: zRoot).status == "files_present", "Z-Image composed shared encoder failed inspection")
        try reject("Z-Image accepted incompatible VAE layout mix") {
            _ = try LibraryComponents.compose(.init(modelID: "z-image-turbo", basePath: url("z").path,
                components: ["vae": url("flux/vae").path]), store: store)
        }
        try reject("Qwen8B encoder accepted for Z-Image Qwen3-4B") {
            _ = try LibraryComponents.compose(.init(modelID: "z-image-turbo", basePath: url("z").path,
                components: ["encoder": url("alternate-encoder.safetensors").path]), store: store)
        }
        try json("bad-index/config.json", ["model_type": "fixture"])
        try tensorFile("bad-index/model.safetensors", ["weight": [1]])
        try json("bad-index/model.safetensors.index.json", ["weight_map": ["weight": "../outside.safetensors"]])
        try reject("Unsafe component shard path accepted") { _ = try ComponentWeightHeader.directory(url("bad-index")) }

        let entriesBeforeRemove = try store.read().installations.count
        try store.unregister(id: snapshot.id)
        try check((try store.read()).installations.count == entriesBeforeRemove - 1, "Unregister did not remove the binding record")
        try check(fm.fileExists(atPath: snapshot.path) && fm.fileExists(atPath: url("encoder-blob").path), "Unregister deleted binding or source bytes")
        try check((try fm.contentsOfDirectory(atPath: store.root.appendingPathComponent("staging").path)).isEmpty, "Composition left staging behind")
        print("PASS: \(assertions) library component assertions; immutable runtime roots, Qwen BF16/GGUF limits, FLUX/Z-Image shapes, source links, rejection/rollback and retained bytes")
    }
}
