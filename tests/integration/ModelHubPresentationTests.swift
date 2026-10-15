import Foundation

/// Pure metadata regression: no files, network, native catalog, GPU or models.
@main struct ModelHubPresentationTests {
    private static func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        guard condition() else { throw LibraryFailure(message: message) }
    }

    static func main() throws {
        let qwen = try model("qwen-image-2.1", name: "Qwen Image 2.1")
        let video = try model("ltx-2.5-distilled", name: "LTX 2.5", video: true)
        let readOnly = try model("future-image", name: "Future image", executor: false)
        let models = [qwen, video, readOnly]
        let first = installation("first", modelID: qwen.id, path: "/models/first-install")
        let second = installation("second", modelID: qwen.id, path: "/models/second-install")
        try localInstallations(models: models, qwen: qwen, first: first, second: second)
        try unifiedSelection(models: models, qwen: qwen, video: video, readOnly: readOnly)
        try searches(models: models, qwen: qwen, first: first, second: second)
        try inspectionStates(models: models, qwen: qwen, first: first, second: second)
        print("PASS model hub metadata: multiple installations/local filtering, unified selection, categories, alternate paths/brand/variant search, strict inspection identity and conservative local states; no file IO/network/GPU")
    }

    private static func localInstallations(models: [StudioModel], qwen: StudioModel,
                                           first: LibraryInstallation, second: LibraryInstallation) throws {
        // Removing the selected first installation clears the draft path. The
        // remaining second installation must stay visible in the local scope.
        let local = ModelHubPresentation.make(models: models, installations: [second], modelPaths: [:], upscalePaths: [:],
                                              selectionID: qwen.id, scope: .local)
        try check(local.entries.count == 1 && local.selectedID == qwen.id &&
                  local.selectedEntry?.localPaths == [second.path] && local.selectedEntry?.localState == .registered,
                  "A remaining installation disappeared after clearing the active path")
        let multiple = ModelHubPresentation.make(models: models, installations: [first, second, second],
            modelPaths: [qwen.id: first.path], upscalePaths: [:], selectionID: qwen.id, scope: .local)
        try check(multiple.entries.count == 1 && multiple.selectedEntry?.installationCount == 2 &&
                  multiple.selectedEntry?.localPaths == [first.path, second.path],
                  "Multiple installations duplicated a model row or its paths")
        let explicit = ModelHubPresentation.make(models: models, installations: [],
            modelPaths: [qwen.id: "/not-checked/legacy-path"], upscalePaths: [:], selectionID: qwen.id, scope: .local)
        try check(explicit.selectedEntry?.localState == .registered && explicit.selectedEntry?.installationCount == 0,
                  "An explicit legacy path was dropped or claimed missing without inspection")
        let none = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
                                             selectionID: qwen.id, scope: .local)
        try check(none.entries.isEmpty && none.selectedID == nil && none.selectedEntry == nil,
                  "Local scope included unconfigured discovery models")
        let upscaler = UpscaleVariant.x2plus
        let upscale = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:],
            upscalePaths: [upscaler.rawValue: "/models/realesrgan.mlpackage"], selectionID: upscaler.modelID, scope: .local)
        try check(upscale.entries.count == 1 && upscale.selectedEntry?.upscaler == upscaler,
                  "Configured upscaler did not appear in local scope")
    }

    private static func unifiedSelection(models: [StudioModel], qwen: StudioModel,
                                          video: StudioModel, readOnly: StudioModel) throws {
        let upscaler = UpscaleVariant.x4plus
        let all = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
                                             selectionID: upscaler.modelID)
        try check(all.entries.count == models.count + UpscaleVariant.allCases.count &&
                  all.selectedID == upscaler.modelID && all.selectedEntry?.model == nil &&
                  all.entries.filter({ $0.id == all.selectedID }).count == 1,
                  "Upscaler selection retained a second selected model")
        try check(all.entries.contains(where: { $0.model?.id == readOnly.id }),
                  "A catalog model without a recipe/executor was hidden")
        let images = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: upscaler.modelID, category: .image)
        try check(images.entries.map(\.id) == [qwen.id, readOnly.id] && images.selectedID == qwen.id,
                  "Image filter retained an invisible selection or included video/upscalers")
        let videos = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: qwen.id, category: .video)
        try check(videos.entries.map(\.id) == [video.id] && videos.selectedEntry?.id == video.id,
                  "Video category did not resolve one visible selection")
        let upscalers = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: qwen.id, category: .upscale)
        try check(upscalers.entries.allSatisfy({ $0.upscaler != nil }) && upscalers.selectedID == UpscaleVariant.allCases.first?.modelID,
                  "Upscale category retained a hidden image model")
    }

    private static func searches(models: [StudioModel], qwen: StudioModel,
                                 first: LibraryInstallation, second: LibraryInstallation) throws {
        let alternate = ModelHubPresentation.make(models: models, installations: [first, second],
            modelPaths: [qwen.id: first.path], upscalePaths: [:], selectionID: qwen.id, query: "SECOND-INSTALL", scope: .local)
        try check(alternate.entries.map(\.id) == [qwen.id], "Search ignored a non-selected installation path")
        let brand = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: nil, query: "  qWEn   阿里  ")
        try check(brand.entries.map(\.id) == [qwen.id], "Brand search lost case/whitespace or Chinese aliases")
        let metadata = [qwen.id: ["Unsloth", "GGUF", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]]
        let variant = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: nil, query: "unsloth GGUF q6_k", searchMetadata: metadata)
        try check(variant.entries.map(\.id) == [qwen.id], "Variant metadata did not participate in discovery search")
        let absent = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: qwen.id, query: "Unsloth GGUF Q7", searchMetadata: metadata)
        try check(absent.entries.isEmpty && absent.selectedID == nil, "Search invented an unsupported quantization variant")
        var registered = second
        registered.repository = "unsloth/Qwen-Image-2.1-GGUF"; registered.provider = "huggingface"
        let provenance = ModelHubPresentation.make(models: models, installations: [registered], modelPaths: [:], upscalePaths: [:],
            selectionID: nil, query: "unsloth huggingface", scope: .local)
        try check(provenance.entries.map(\.id) == [qwen.id], "Local search ignored repository/provider provenance")
    }

    private static func inspectionStates(models: [StudioModel], qwen: StudioModel,
                                         first: LibraryInstallation, second: LibraryInstallation) throws {
        var broken = InstallationInspection(modelID: qwen.id, path: first.path)
        broken.checkedAt = Date(timeIntervalSince1970: 10)
        broken.issues = [.init(code: "missing_file", path: "processor/tokenizer.json", message: "Missing tokenizer")]
        let selectedBroken = ModelHubPresentation.make(models: models, installations: [first, second],
            modelPaths: [qwen.id: first.path], upscalePaths: [:], selectionID: qwen.id, inspections: ["arbitrary-key": broken])
        try check(selectedBroken.selectedEntry?.localState == .missing,
                  "A checked broken active installation was presented as merely uninspected")
        let unselectedBroken = ModelHubPresentation.make(models: models, installations: [first, second], modelPaths: [:],
            upscalePaths: [:], selectionID: qwen.id, inspections: ["arbitrary-key": broken])
        try check(unselectedBroken.selectedEntry?.localState == .registered,
                  "A broken alternative masked a still-unchecked installation")
        let onlyBroken = ModelHubPresentation.make(models: models, installations: [first], modelPaths: [:], upscalePaths: [:],
            selectionID: qwen.id, scope: .local, inspections: ["arbitrary-key": broken])
        try check(onlyBroken.selectedEntry?.localState == .missing && onlyBroken.entries.count == 1,
                  "Local scope hid the broken installation instead of allowing repair")
        var repaired = InstallationInspection(modelID: qwen.id, path: first.path)
        repaired.checkedAt = Date(timeIntervalSince1970: 20)
        let newer = ModelHubPresentation.make(models: models, installations: [first], modelPaths: [qwen.id: first.path], upscalePaths: [:],
            selectionID: qwen.id, inspections: ["old": broken, "new": repaired])
        try check(newer.selectedEntry?.localState == .registered, "An older failure overrode a newer bounded file check")
        var wrongIdentity = broken
        wrongIdentity.modelID = "other-model"
        let unrelated = ModelHubPresentation.make(models: models, installations: [second], modelPaths: [qwen.id: second.path], upscalePaths: [:],
            selectionID: qwen.id, inspections: [qwen.id + "\n" + second.path: wrongIdentity])
        try check(unrelated.selectedEntry?.localState == .registered,
                  "A mismatched model/path report was trusted because its dictionary key looked correct")
        let wrongPath = ModelHubPresentation.make(models: models, installations: [second], modelPaths: [qwen.id: second.path], upscalePaths: [:],
            selectionID: qwen.id, inspections: [qwen.id + "\n" + second.path: broken])
        try check(wrongPath.selectedEntry?.localState == .registered,
                  "Another installation's failure was assigned to the configured path")
        let unconfigured = ModelHubPresentation.make(models: models, installations: [], modelPaths: [:], upscalePaths: [:],
            selectionID: qwen.id, inspections: ["stale": broken])
        try check(unconfigured.selectedEntry?.localState == .unconfigured,
                  "A stale inspection created a phantom local installation")
    }

    private static func model(_ id: String, name: String, video: Bool = false, executor: Bool = true) throws -> StudioModel {
        let value: [String: Any] = ["id": id, "name": name, "executor": executor,
            "output": video ? "video" : "image", "operations": [video ? "video.generate" : "image.generate"],
            "default_steps": 25, "default_frames": 1, "default_width": 512, "default_height": 512]
        return try JSONDecoder().decode(StudioModel.self, from: JSONSerialization.data(withJSONObject: value))
    }

    private static func installation(_ id: String, modelID: String, path: String) -> LibraryInstallation {
        LibraryInstallation(id: id, modelID: modelID, name: "Installation \(id)", path: path, managed: false,
            provider: nil, repository: nil, revision: nil, manifest: nil, components: [:], createdAt: Date(timeIntervalSince1970: 1))
    }
}
