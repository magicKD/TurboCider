import Foundation

extension StudioDraft {
    var resolvedPrunaVAEPath: String? {
        let path = ltxPrunaVAEPath.isEmpty
            ? modelPaths["ltx-2.5-distilled"].map {
                URL(fileURLWithPath: $0).appendingPathComponent("vae/prunavaed.safetensors").path
            } : ltxPrunaVAEPath
        guard let path, path.hasPrefix("/"),
              let values = try? URL(fileURLWithPath: path).resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true, values.fileSize == Int(PrunaVAEAsset.model.size) else { return nil }
        return path
    }
    func validateLTXVideoVAE() throws {
        guard ["conv", "prunavaed"].contains(ltxVideoVAE) else {
            throw NativeFailure(message: "LTX 视频 VAE 选择无效。")
        }
        if ltxVideoVAE == "prunavaed" {
            guard !usesPublicStreaming else {
                throw NativeFailure(message: "PrunaVAED 尚未适配校准内存档位，请先关闭内存档位。")
            }
            guard resolvedPrunaVAEPath != nil else {
                throw NativeFailure(message: "请先在模型库的 VAE 模型中下载或选择 PrunaVAED。")
            }
        }
    }
}
import ImageIO

extension TinyVAEAsset {
    func bundledPath(in resources: URL? = Bundle.main.resourceURL) -> String? {
        guard let url = resources?.appendingPathComponent("tiny-vae").appendingPathComponent(filename),
              (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
            return nil
        }
        return url.path
    }
}

extension StudioDraft {
    var supportsTinyVAE: Bool {
        ["qwen-image-2.1", "z-image-turbo", "z-image-turbo-gguf"].contains(modelID)
    }
    var tinyArchitecture: String { modelID == "qwen-image-2.1" ? "taeqi2_1" : "taef1" }
    var resolvedTinyVAEPath: String? {
        if let explicit = tinyVAEPaths[tinyArchitecture], !explicit.isEmpty { return explicit }
        if let asset = TinyVAEAsset.all.first(where: { $0.id == tinyArchitecture }),
           let bundled = asset.bundledPath() { return bundled }
        guard !modelPath.isEmpty else { return nil }
        let name = tinyArchitecture + ".safetensors"
        let model = URL(fileURLWithPath: modelPath)
        return [model, model.deletingLastPathComponent()].map {
            $0.appendingPathComponent("tiny-vae").appendingPathComponent(name).path
        }.first {
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

enum TinyPreviewFrame {
    static func matches(_ data: Data, width: Int, height: Int) -> Bool {
        guard width > 0, height > 0,
              data.starts(with: [137, 80, 78, 71, 13, 10, 26, 10]),
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let actualWidth = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let actualHeight = properties[kCGImagePropertyPixelHeight] as? NSNumber else { return false }
        return actualWidth.intValue == width && actualHeight.intValue == height
    }
}
