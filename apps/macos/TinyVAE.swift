import Foundation
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
        isWan22Video || ["qwen-image-2.1", "z-image-turbo", "z-image-turbo-gguf"].contains(modelID)
    }
    var tinyArchitecture: String {
        if isWan22Video { return "taew2_2_super" }
        return modelID == "qwen-image-2.1" ? "taeqi2_1" : "taef1"
    }
    var resolvedTinyVAEPath: String? { resolveTinyVAEPath(resources: Bundle.main.resourceURL) }
    func resolveTinyVAEPath(resources: URL?) -> String? {
        if let explicit = tinyVAEPaths[tinyArchitecture], !explicit.isEmpty { return explicit }
        if isWan22Video,
           let bundled = resources?.appendingPathComponent("tiny-vae").appendingPathComponent("taew2_2_super.safetensors"),
           (try? bundled.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            return bundled.path
        }
        if let asset = TinyVAEAsset.all.first(where: { $0.id == tinyArchitecture }),
           let bundled = asset.bundledPath(in: resources) { return bundled }
        guard !modelPath.isEmpty else { return nil }
        let name = tinyArchitecture + ".safetensors"
        let model = URL(fileURLWithPath: modelPath)
        if isWan22Video {
            let candidate = model.appendingPathComponent("vae").appendingPathComponent(name).path
            if FileManager.default.fileExists(atPath: candidate) { return candidate }
        }
        return [model, model.deletingLastPathComponent()].map {
            $0.appendingPathComponent("tiny-vae").appendingPathComponent(name).path
        }.first {
            FileManager.default.fileExists(atPath: $0)
        }
    }
    func tinyVAEConfiguration() throws -> NativeTinyVAEConfig? {
        guard supportsTinyVAE else { return nil }
        if isWan22Video {
            guard ["original", "tiny"].contains(wan22TinyVAEDecoder) else {
                throw NativeFailure(message: "Wan2.2 视频解码设置无效。")
            }
            guard wan22TinyVAEDecoder == "tiny" else { return nil }
            guard streaming.selection == .off else {
                throw NativeFailure(message: "TAEW2.2-Super 尚未标定流式内存档位，请先关闭内存档位。")
            }
            guard let path = resolvedTinyVAEPath, path.hasPrefix("/"),
                  FileManager.default.fileExists(atPath: path) else {
                throw NativeFailure(message: "请先选择 taew2_2_super.safetensors；不会自动回退为完整 VAE。")
            }
            return NativeTinyVAEConfig(path: path, encoder: "original", decoder: "tiny", preview: false, interval: 2)
        }
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
