import Foundation

enum Qwen21WeightCacheBudget: String, CaseIterable {
    case minimum, gib1, gib2, gib4, gib6, gib8, custom
    var title: String {
        switch self {
        case .minimum: return "最低占用（按层读取）"
        case .custom: return "自定义"
        default: return "\(fixedMiB / 1024) GiB"
        }
    }
    var fixedMiB: Int {
        switch self {
        case .gib1: return 1024
        case .gib2: return 2048
        case .gib4: return 4096
        case .gib6: return 6144
        case .gib8: return 8192
        default: return 0
        }
    }
}

struct Qwen21Installation {
    static let modelID = "qwen-image-2.1"
    static let turboID = "qwen-image-2.1-turbo"
    static let turboPresetID = "turbo-int8-convrot"
    static let turboName = "Qwen Image 2.1 Turbo"

    struct Choice: Identifiable {
        let path: String
        var title: String
        var id: String { path }
    }

    // Match the native source priority even when a directory contains both versions.
    static func checkpointName(at path: String) -> String? {
        guard !path.isEmpty else { return nil }
        let root = URL(fileURLWithPath: path).appendingPathComponent("diffusion_models")
        return ["qwen_image_2.1_bf16.safetensors", "qwen_image_2.1_int8_convrot.safetensors",
                "qwen_image_2.1_turbo_bf16.safetensors", "qwen_image_2.1_turbo_int8_convrot.safetensors",
                "qwen-image-2.1-Q4_K_M.gguf"].first {
            FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path)
        }
    }

    static func isTurbo(at path: String) -> Bool {
        checkpointName(at: path)?.hasPrefix("qwen_image_2.1_turbo_") == true
    }

    static func displayName(at path: String) -> String {
        guard checkpointName(at: path) != nil else { return "Qwen Image 2.1 / Turbo" }
        return isTurbo(at: path) ? turboName : "Qwen Image 2.1"
    }

    static func title(at path: String) -> String {
        guard let name = checkpointName(at: path) else { return displayName(at: path) }
        let precision = name.contains("int8_convrot") ? "INT8 ConvRot" : name.hasSuffix(".gguf") ? "GGUF Q4_K_M" : "BF16"
        return displayName(at: path) + " · " + precision + (isTurbo(at: path) ? " · 8 步" : "")
    }

    static func choices(_ installations: [LibraryInstallation], currentPath: String) -> [Choice] {
        let entries = installations.filter { $0.modelID == modelID }.map(\.path) + (currentPath.isEmpty ? [] : [currentPath])
        let current = currentPath.isEmpty ? "" : URL(fileURLWithPath: currentPath).resolvingSymlinksInPath().path
        var seen = Set<String>()
        var result: [Choice] = []
        for path in entries {
            let canonical = URL(fileURLWithPath: path).resolvingSymlinksInPath().path
            guard seen.insert(canonical).inserted else { continue }
            result.append(Choice(path: canonical == current ? currentPath : path, title: title(at: path)))
        }
        let counts = Dictionary(grouping: result, by: \.title).mapValues(\.count)
        var positions: [String: Int] = [:]
        return result.map { choice in
            var choice = choice
            if counts[choice.title, default: 0] > 1 {
                positions[choice.title, default: 0] += 1
                choice.title += " · 安装 \(positions[choice.title]!)"
            }
            return choice
        }
    }
}
