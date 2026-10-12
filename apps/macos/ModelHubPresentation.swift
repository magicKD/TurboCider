import Foundation

enum ModelHubScope: String, CaseIterable, Identifiable {
    case discover, local
    var id: String { rawValue }
    var title: String { self == .discover ? "发现" : "本机" }
}

enum ModelHubCategory: String, CaseIterable, Identifiable {
    case all, image, video, upscale
    var id: String { rawValue }
    var title: String {
        switch self {
        case .all: return "全部"
        case .image: return "图像"
        case .video: return "视频"
        case .upscale: return "超分"
        }
    }
}

/// Registration and bounded file checks are not proof of a runnable model.
enum ModelHubLocalState: Equatable {
    case unconfigured, registered, missing
    var title: String {
        switch self {
        case .unconfigured: return "未配置"
        case .registered: return "已登记 · 待检查"
        case .missing: return "检查发现缺失或无效"
        }
    }
}

struct ModelHubEntry: Identifiable {
    let id: String
    let model: StudioModel?
    let upscaler: UpscaleVariant?
    let localPaths: [String]
    let installationCount: Int
    let localState: ModelHubLocalState
    var title: String { model?.displayName ?? upscaler?.title ?? id }
}

/// Pure display metadata. No filesystem checks, network requests, model catalog
/// calls, draft mutation, source hashing or tensor loading happen here.
struct ModelHubPresentation {
    let entries: [ModelHubEntry]
    let selectedID: String?
    var selectedEntry: ModelHubEntry? { entries.first { $0.id == selectedID } }

    static func make(models: [StudioModel], installations: [LibraryInstallation],
                     modelPaths: [String: String], upscalePaths: [String: String],
                     selectionID: String?, query: String = "",
                     category: ModelHubCategory = .all, scope: ModelHubScope = .discover,
                     inspections: [String: InstallationInspection] = [:],
                     searchMetadata: [String: [String]] = [:]) -> Self {
        let tokens = query.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        var seenIDs = Set<String>()
        let upscalerIDs = Set(UpscaleVariant.allCases.map(\.modelID))
        var entries: [ModelHubEntry] = []

        func append(id: String, model: StudioModel?, upscaler: UpscaleVariant?, currentPath: String?) {
            guard seenIDs.insert(id).inserted else { return }
            let registered = installations.filter { $0.modelID == id }
            var paths: [String] = []
            for path in [currentPath].compactMap({ $0 }) + registered.map(\.path) {
                guard !path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !paths.contains(path) else { continue }
                paths.append(path)
            }
            let localState = state(modelID: id, paths: paths, currentPath: currentPath, inspections: inspections)
            guard scope != .local || localState != .unconfigured else { return }
            switch category {
            case .all: break
            case .image: guard model?.isVideo == false else { return }
            case .video: guard model?.isVideo == true else { return }
            case .upscale: guard upscaler != nil else { return }
            }
            var terms = [id, model?.name ?? upscaler?.title ?? ""] + paths + (searchMetadata[id] ?? [])
            terms += registered.flatMap { [$0.name, $0.path, $0.provider ?? "", $0.repository ?? "", $0.revision ?? ""] }
            terms += aliases(for: id)
            terms.append(upscaler != nil ? "超分 upscale upscaler Real-ESRGAN" : model?.isVideo == true ? "视频 video" : "图像 image")
            let searchable = terms.joined(separator: " ")
            guard tokens.allSatisfy({ searchable.localizedCaseInsensitiveContains($0) }) else { return }
            entries.append(ModelHubEntry(id: id, model: model, upscaler: upscaler, localPaths: paths,
                                         installationCount: Set(registered.map(\.path).filter { !$0.isEmpty }).count,
                                         localState: localState))
        }

        for model in models where !upscalerIDs.contains(model.id) {
            append(id: model.id, model: model, upscaler: nil, currentPath: modelPaths[model.id])
        }
        for upscaler in UpscaleVariant.allCases {
            append(id: upscaler.modelID, model: nil, upscaler: upscaler,
                   currentPath: upscalePaths[upscaler.rawValue] ?? modelPaths[upscaler.modelID])
        }
        // One resolved selection drives both row highlighting and details.
        let selectedID = entries.contains(where: { $0.id == selectionID }) ? selectionID : entries.first?.id
        return Self(entries: entries, selectedID: selectedID)
    }

    private static func state(modelID: String, paths: [String], currentPath: String?,
                              inspections: [String: InstallationInspection]) -> ModelHubLocalState {
        guard !paths.isEmpty else { return .unconfigured }
        func report(for path: String) -> InstallationInspection? {
            inspections.values.filter { $0.modelID == modelID && $0.path == path }
                .max { $0.checkedAt < $1.checkedAt }
        }
        if let currentPath, paths.contains(currentPath) {
            return report(for: currentPath)?.issues.isEmpty == false ? .missing : .registered
        }
        // With no active path, another unchecked or complete installation is
        // still useful. A broken alternative must not hide that installation.
        return paths.allSatisfy { report(for: $0)?.issues.isEmpty == false } ? .missing : .registered
    }

    private static func aliases(for id: String) -> [String] {
        if id.hasPrefix("qwen") { return ["Qwen Alibaba 阿里 通义 千问"] }
        if id.hasPrefix("flux") { return ["FLUX Black Forest Labs BFL"] }
        if id.hasPrefix("z-image") { return ["Z-Image Tongyi Alibaba 阿里 通义"] }
        if id.hasPrefix("ltx") { return ["LTX Lightricks"] }
        if id.hasPrefix("minimax") { return ["MiniMax H3"] }
        if id.hasPrefix("wan") { return ["Wan Alibaba 阿里 万相"] }
        return []
    }
}
