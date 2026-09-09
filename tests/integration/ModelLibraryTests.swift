import Foundation

@main struct ModelLibraryTests {
    static func main() throws {
        func check(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
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
