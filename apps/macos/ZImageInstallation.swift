import Foundation

/// App-owned directory bindings keep user checkpoints in their original location.
struct ZImageInstallation {
    static func splitDirectory(_ model: URL) -> URL? {
        [model.appendingPathComponent("split_files"), model.appendingPathComponent("models"), model]
            .first { (FileManager.default.fileExists(atPath: $0.appendingPathComponent("diffusion_models/z_image_turbo_bf16.safetensors").path)
                || FileManager.default.fileExists(atPath: $0.appendingPathComponent("diffusion_models/z_image_turbo_int8_convrot.safetensors").path)
                || FileManager.default.fileExists(atPath: $0.appendingPathComponent("diffusion_models/z_image_turbo_nvfp4.safetensors").path))
                && FileManager.default.fileExists(atPath: $0.appendingPathComponent("vae/ae.safetensors").path) }
    }
    static func hasSharedText(_ model: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: model.appendingPathComponent("tokenizer/tokenizer.json").path)
            && ((try? fm.contentsOfDirectory(atPath: model.appendingPathComponent("text_encoder").path)) ?? []).contains { $0.hasSuffix(".safetensors") }
    }
    static func needsSharedText(_ model: URL) -> Bool {
        guard let split = splitDirectory(model) else { return false }
        return !hasSharedText(model) && !(FileManager.default.fileExists(atPath: split.appendingPathComponent("text_encoders/qwen_3_4b.safetensors").path)
            && FileManager.default.fileExists(atPath: model.appendingPathComponent("tokenizer/tokenizer.json").path))
    }
    static func install(model: URL, sharedText: URL?, directory: URL) throws -> URL {
        guard let split = splitDirectory(model) else {
            guard hasSharedText(model), FileManager.default.fileExists(atPath: model.appendingPathComponent("transformer").path),
                  FileManager.default.fileExists(atPath: model.appendingPathComponent("vae").path) else {
                throw NativeFailure(message: "请选择完整的 Z-Image 模型目录，包含 diffusion_models/vae 或 transformer/vae。")
            }
            return model
        }
        let text = sharedText ?? model
        if sharedText != nil || needsSharedText(model) {
            guard hasSharedText(text) else { throw NativeFailure(message: "共享文本模型需包含 tokenizer/tokenizer.json 和 text_encoder 权重，例如 FLUX.2-klein-4B。") }
        }
        let fm = FileManager.default
        let target = directory.appendingPathComponent("z-image-\(UUID().uuidString)")
        try fm.createDirectory(at: target, withIntermediateDirectories: true)
        do {
            try fm.createSymbolicLink(at: target.appendingPathComponent("split_files"), withDestinationURL: split.resolvingSymlinksInPath())
            try fm.createSymbolicLink(at: target.appendingPathComponent("tokenizer"), withDestinationURL: text.appendingPathComponent("tokenizer").resolvingSymlinksInPath())
            if hasSharedText(text) {
                try fm.createSymbolicLink(at: target.appendingPathComponent("text_encoder"), withDestinationURL: text.appendingPathComponent("text_encoder").resolvingSymlinksInPath())
            }
            let provenance = ["model": model.path, "shared_text": text.path]
            try JSONEncoder().encode(provenance).write(to: target.appendingPathComponent("installation.json"), options: .atomic)
            return target
        } catch {
            try? fm.removeItem(at: target)
            throw error
        }
    }
}
