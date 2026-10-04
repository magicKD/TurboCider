import Foundation

// Export the real Playground role/prompt/request pipeline for external native
// acceptance. This executable never plans a request or loads model weights.
@main struct PlaygroundRequestExport {
    struct Configuration: Decodable {
        let template: PlaygroundTemplate
        let modelPath: String
        let loraPath: String?
        let personPath: String?
        let sourcePath: String?
        let targetPath: String?
        let expansion: Double?
        let clothingPath: String?
        let scenePath: String?
        let stateDirectory: String
        let outputPath: String
        let referenceSize: Int?
        let steps: Int?
        let seed: Int?
        let residency: String?
        let instruction: String?
    }

    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            throw NativeFailure(message: "Usage: playground-request-export configuration.json")
        }
        let configuration = try JSONDecoder().decode(Configuration.self,
            from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        guard configuration.stateDirectory.hasPrefix("/"), configuration.outputPath.hasPrefix("/"),
              configuration.modelPath.hasPrefix("/") else {
            throw NativeFailure(message: "Use explicit absolute model, image, isolated state and output paths.")
        }
        let directory = URL(fileURLWithPath: configuration.stateDirectory, isDirectory: true)
        // Never overwrite a user draft or silently reuse inputs from an earlier
        // export. Each acceptance case must provide a fresh isolated directory.
        if FileManager.default.fileExists(atPath: directory.path),
           !(try FileManager.default.contentsOfDirectory(atPath: directory.path)).isEmpty {
            throw NativeFailure(message: "The isolated state directory must be new or empty.")
        }
        var settings = StudioDraft()
        settings.modelID = "qwen-image-2.1"
        settings.modelPaths[settings.modelID] = configuration.modelPath
        settings.operation = "image.edit"; settings.width = 512; settings.height = 512
        settings.steps = configuration.steps ?? (configuration.loraPath == nil ? 25 : 6)
        settings.seedText = String(configuration.seed ?? 42); settings.randomSeed = false
        settings.acceleration = StudioAcceleration(policy: "gpu")
        settings.qwen21ReferenceSize = configuration.referenceSize ?? 512
        settings.qwen21DiTCache = "off"; settings.promptEnhance = false
        settings.residency = configuration.residency ?? "component_staged"
        settings.loraStrategy = "inference_time"
        if let path = configuration.loraPath { settings.loras = [StudioLoRA(path: path, strength: 1)] }
        // Metadata registry only. No NativeEngine instance, plan or generate.
        let state = PlaygroundState(directory: directory, initialSettings: settings, models: StudioModel.catalog())
        state.selectTemplate(configuration.template)
        if let instruction = configuration.instruction { state.setInstruction(instruction) }
        var paths: [String] = []
        if [.outfit, .identity, .face].contains(configuration.template) {
            guard let person = configuration.personPath, person.hasPrefix("/") else {
                throw NativeFailure(message: "This workflow requires an absolute personPath.")
            }
            paths.append(person)
        }
        switch configuration.template {
        case .outfit:
            guard let clothing = configuration.clothingPath else { throw NativeFailure(message: "Outfit requires clothingPath.") }
            paths.append(clothing)
        case .identity:
            if let scene = configuration.scenePath { paths.append(scene) }
        case .face:
            guard let target = configuration.targetPath else { throw NativeFailure(message: "Face requires targetPath.") }
            paths.append(target)
        case .outpaint:
            guard let source = configuration.sourcePath else { throw NativeFailure(message: "Outpaint requires sourcePath.") }
            paths.append(source)
            guard state.setOutpaintExpansion(configuration.expansion ?? 1.5) else {
                throw NativeFailure(message: "Invalid expansion.")
            }
        case .transparent:
            if let source = configuration.sourcePath { paths.append(source) }
        }
        guard paths.allSatisfy({ $0.hasPrefix("/") }) else { throw NativeFailure(message: "Reference paths must be absolute.") }
        if !paths.isEmpty, !(await state.importFiles(paths.map { URL(fileURLWithPath: $0) })) {
            throw NativeFailure(message: state.message ?? "Cannot import Playground references.")
        }
        let draft = try state.generationDraft()
        let request = try draft.request(output: URL(fileURLWithPath: configuration.outputPath))
        state.recordSubmission(draft, seed: request.seed, template: configuration.template)
        state.save()
        guard state.saved else { throw NativeFailure(message: state.storageError ?? "Cannot save the isolated Playground draft.") }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let requestData = try encoder.encode(request)
        try requestData.write(to: directory.appendingPathComponent("request.json"), options: .withoutOverwriting)
        try encoder.encode(NativeRequestV2(legacy: request)).write(to: directory.appendingPathComponent("request-v2.json"), options: .withoutOverwriting)
        try encoder.encode(draft).write(to: directory.appendingPathComponent("generation-draft.json"), options: .withoutOverwriting)
        print(String(decoding: requestData, as: UTF8.self))
    }
}
