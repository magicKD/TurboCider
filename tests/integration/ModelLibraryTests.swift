import Foundation

@main struct ModelLibraryTests {
    @MainActor static func main() throws {
        func check(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tc-model-switch-\(UUID())")
        defer { try? FileManager.default.removeItem(at: directory) }
        let studio = StudioState(directory: directory)
        studio.selectModel("z-image-turbo")
        studio.draft.loras = [StudioLoRA(path: "/fixture/z.safetensors", strength: 0.6)]
        studio.selectModel("flux2-klein-4b")
        try check(studio.draft.loras.isEmpty, "Z-Image LoRA leaked into FLUX")
        studio.selectModel("z-image-turbo")
        try check(studio.draft.loras.first?.strength == 0.6, "Model switch lost LoRA settings")
        studio.save()
        let restored = StudioState(directory: directory)
        restored.selectModel("flux2-klein-4b")
        restored.selectModel("z-image-turbo")
        try check(restored.draft.loras.first?.path == "/fixture/z.safetensors", "LoRA association did not survive restart")
        let fm = FileManager.default
        let source = directory.appendingPathComponent("source")
        for folder in ["models/diffusion_models", "models/vae", "text_encoder", "tokenizer"] {
            try fm.createDirectory(at: source.appendingPathComponent(folder), withIntermediateDirectories: true)
        }
        for file in ["models/diffusion_models/z_image_turbo_bf16.safetensors", "models/vae/ae.safetensors", "text_encoder/model.safetensors", "tokenizer/tokenizer.json"] {
            try Data("fixture".utf8).write(to: source.appendingPathComponent(file))
        }
        let oldParent = directory.appendingPathComponent("old-output")
        let old = try ZImageInstallation.install(model: source, sharedText: nil, directory: oldParent)
        let migrated = try ZImageInstallation.install(model: old, sharedText: nil, directory: directory.appendingPathComponent("library/bindings"))
        try fm.removeItem(at: oldParent)
        try check(ZImageInstallation.splitDirectory(migrated) != nil && ZImageInstallation.hasSharedText(migrated), "Migrated binding still depends on old output")
        try fm.removeItem(at: source.appendingPathComponent("models/diffusion_models/z_image_turbo_bf16.safetensors"))
        let inspection = InstallationInspection.inspect(modelID: "z-image-turbo", root: migrated)
        try check(!inspection.issues.isEmpty, "Broken weight link was not diagnosed")
        let catalog = StudioModel.catalog()
        guard let ltx = catalog.first(where: { $0.id == "ltx-2.5-distilled" }),
              let z = catalog.first(where: { $0.id == "z-image-turbo" }),
              let flux = catalog.first(where: { $0.id == "flux2-klein-4b" }),
              let h3 = catalog.first(where: { $0.id == "minimax-h3-turbo" }) else {
            throw NativeFailure(message: "Native catalog is incomplete")
        }
        try check(!ltx.acceptsImageInputs && ltx.availableOperations == ["video.generate"], "LTX browser leaked unqualified image operations")
        try check(!z.acceptsImageInputs && z.availableOperations == ["image.generate"], "Z-Image browser accepts images")
        try check(flux.acceptsImageInputs && h3.acceptsImageInputs, "Validated reference inputs are hidden")
        try check(!ltx.canGenerateAudio && !z.canGenerateAudio && !flux.canGenerateAudio && h3.canGenerateAudio, "Audio gate differs from executable model capabilities")
        try check(z.matchesLibrarySearch("Z-IMAGE turbo", path: ""), "Case-insensitive model search failed")
        try check(z.matchesLibrarySearch("文字", path: ""), "Capability search failed")
        try check(ltx.matchesLibrarySearch("视频", path: ""), "Video search failed")
        try check(z.matchesLibrarySearch("external Models", path: "/external/Models/z"), "Path search failed")
        try check(!z.matchesLibrarySearch("Z-Image nonexistent", path: ""), "Search words were not intersected")
        try check(catalog.allSatisfy { $0.matchesLibrarySearch(" \n ", path: "") }, "Blank search should show all entries")
        let unavailable = StudioModel(id: "fixture", name: "Fixture", executor: false, output: "video", operations: ["video.image"], default_steps: 1, default_frames: 1, default_width: 64, default_height: 64)
        try check(!unavailable.acceptsImageInputs && unavailable.availableOperations.isEmpty, "Disabled executor advertised operations")
        print("PASS: catalog capability gates, text/image distinction, name/path/multilingual search")
    }
}
