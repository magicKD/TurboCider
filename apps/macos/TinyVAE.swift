import Foundation

extension StudioDraft {
    var supportsTinyVAE: Bool {
        ["qwen-image-2.1", "z-image-turbo", "z-image-turbo-gguf"].contains(modelID)
    }
    var tinyArchitecture: String { modelID == "qwen-image-2.1" ? "taeqi2_1" : "taef1" }
    var resolvedTinyVAEPath: String? {
        if let explicit = tinyVAEPaths[tinyArchitecture], !explicit.isEmpty { return explicit }
        let name = tinyArchitecture + ".safetensors"
        var roots = [Bundle.main.resourceURL?.appendingPathComponent("tiny-vae")]
        if !modelPath.isEmpty {
            let model = URL(fileURLWithPath: modelPath)
            roots += [
                model.appendingPathComponent("tiny-vae"),
                model.deletingLastPathComponent().appendingPathComponent("tiny-vae"),
            ]
        }
        return roots.compactMap { $0?.appendingPathComponent(name).path }.first {
            FileManager.default.fileExists(atPath: $0)
        }
    }
    func tinyVAEConfiguration() throws -> NativeTinyVAEConfig? {
        guard supportsTinyVAE else { return nil }
        guard ["original", "tiny"].contains(tinyVAEEncoder),
            ["original", "tiny"].contains(tinyVAEDecoder),
            (1...50).contains(tinyPreviewInterval)
        else { throw NativeFailure(message: "Tiny VAE 设置无效。预览间隔需要为 1–50 步。") }
        // Z-Image is text-to-image; no reference encoder is invoked on that route.
        let encoder =
            modelID == "qwen-image-2.1" && operation == "image.edit" ? tinyVAEEncoder : "original"
        guard encoder == "tiny" || tinyVAEDecoder == "tiny" || tinyPreview else { return nil }
        guard streaming.selection == .off else {
            throw NativeFailure(message: "Tiny VAE／实时预览尚未标定流式内存档位，请先关闭内存档位。")
        }
        guard let path = resolvedTinyVAEPath, path.hasPrefix("/"),
            FileManager.default.fileExists(atPath: path)
        else {
            throw NativeFailure(message: "请先在 Tiny VAE 设置中选择 \(tinyArchitecture).safetensors 模型。")
        }
        return NativeTinyVAEConfig(
            path: path, encoder: encoder, decoder: tinyVAEDecoder,
            preview: tinyPreview, interval: tinyPreviewInterval)
    }
}

/// Preview telemetry never authorizes publication of a final artifact. Only
/// the exact request-owned sidecar can be read, not arbitrary worker paths.
enum TinyPreviewPath {
    static func expected(output: String) -> String { output + ".preview.png" }
    static func valid(_ event: NativeEvent, output: String, steps: Int) -> Bool {
        event.phase == "preview" && event.preview_path == expected(output: output)
            && event.completed > 0 && event.completed <= steps && event.total == steps
    }
}
