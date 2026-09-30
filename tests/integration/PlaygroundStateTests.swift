import AppKit
import Combine
import Foundation
import ImageIO
import UniformTypeIdentifiers

@MainActor private final class DeferredPlaygroundImageProvider {
    private var reply: ((Data?, Error?) -> Void)?
    func makeProvider() -> NSItemProvider {
        let provider = NSItemProvider()
        provider.registerDataRepresentation(forTypeIdentifier: UTType.png.identifier, visibility: .all) { reply in
            Task { @MainActor in self.reply = reply }
            return nil
        }
        return provider
    }
    func waitUntilRequested() async throws {
        let start = ContinuousClock.now
        while reply == nil {
            guard start.duration(to: .now) < .seconds(3) else {
                throw NativeFailure(message: "Deferred Playground image provider was not requested")
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
    func finish(_ data: Data) { reply?(data, nil); reply = nil }
}

// CPU-only state, persistence and immutable-input contracts. No model catalog,
// native request planning or generation is invoked by this target.
@main struct PlaygroundStateTests {
    private static func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try condition() else { throw NativeFailure(message: message) }
    }

    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-playground-state-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = try modelFixture()
        let person = root.appendingPathComponent("person.png")
        let clothing = root.appendingPathComponent("clothing.png")
        let scene = root.appendingPathComponent("scene.png")
        try writePNG(to: person, width: 64, height: 96, color: CGColor(red: 0.8, green: 0.2, blue: 0.2, alpha: 1))
        try writePNG(to: clothing, width: 80, height: 64, color: CGColor(red: 0.2, green: 0.8, blue: 0.2, alpha: 1))
        try writePNG(to: scene, width: 96, height: 64, color: CGColor(red: 0.2, green: 0.2, blue: 0.8, alpha: 1))

        var initial = StudioDraft()
        initial.modelID = model.id; initial.modelPaths[model.id] = root.appendingPathComponent("model-fixture").path
        initial.steps = 25; initial.seedText = "137"; initial.prompt = "Creation-only prompt"
        initial.acceleration = StudioAcceleration(); initial.loraStrategy = "inference_time"
        // Validation checks file presence; this placeholder is never loaded as weights.
        let adapter = root.appendingPathComponent("ordinary-fixture.safetensors")
        try Data("not model weights".utf8).write(to: adapter)
        initial.loras = [StudioLoRA(path: adapter.path, strength: 0.6)]
        let creation = StudioState(directory: root.appendingPathComponent("creation"), models: [model])
        creation.draft = initial
        let creationAsset = try await creation.importer.importFile(person)
        creation.draft.assets = [creationAsset]; creation.draft.initImageID = creationAsset.id
        creation.save()
        let creationBefore = try bytes(creation.draft)
        let creationFileBefore = try Data(contentsOf: root.appendingPathComponent("creation/studio-draft.json"))

        let stateDirectory = root.appendingPathComponent("playground")
        let state = try await draftIsolationAndRoles(directory: stateDirectory, creation: creation,
            model: model, person: person, clothing: clothing, scene: scene)
        try check(try bytes(creation.draft) == creationBefore, "Playground changed the Creation draft")
        try check(try Data(contentsOf: root.appendingPathComponent("creation/studio-draft.json")) == creationFileBefore,
                  "Playground wrote the Creation persistence file")
        try await failedImportRollback(state, directory: stateDirectory, person: person, root: root)
        try await resultOwnership(state, directory: stateDirectory, model: model, initial: initial, output: scene)
        try damagedFileProtection(root: root, model: model, initial: initial, validDocument: state.document)
        try await preprocessingAndPending(root: root, model: model, initial: initial, clothing: clothing)
        print("PASS Playground independent drafts/roles/import rollback/persistence/provenance, source-based resize/restore, parameter blocks and provider pending/cancel/context guards (CPU only)")
    }

    @MainActor private static func draftIsolationAndRoles(directory: URL, creation: StudioState,
        model: StudioModel, person: URL, clothing: URL, scene: URL) async throws -> PlaygroundState {
        let state = PlaygroundState(directory: directory, initialSettings: creation.draft, models: [model])
        for template in PlaygroundTemplate.allCases {
            let settings = state.document.templates[template.rawValue]!.settings
            try check(settings.prompt.isEmpty && settings.assets.isEmpty && settings.initImageID == nil && settings.operation == "image.edit",
                      "Initial Playground settings inherited Creation content")
            try check(settings.modelPath == creation.draft.modelPath && settings.steps == 25 && settings.loras == creation.draft.loras &&
                      settings.width == creation.draft.width && settings.height == creation.draft.height &&
                      settings.seedText == creation.draft.seedText && settings.acceleration?.policy == "gpu" &&
                      settings.loraStrategy == creation.draft.loraStrategy,
                      "Initial Playground settings lost generation preferences")
        }
        state.setInstruction("Outfit-specific instruction")
        let groupImported = await state.importFiles([person, clothing])
        try check(groupImported && state.asset(for: .person)?.name == person.lastPathComponent &&
                  state.asset(for: .clothing)?.name == clothing.lastPathComponent,
                  "Group import did not fill person then clothing")
        let outfitAssets = state.orderedAssets
        var copiedSettings = creation.draft
        copiedSettings.steps = 30; copiedSettings.seedText = "251"
        try check(state.syncSettings(from: copiedSettings), "Compatible Creation settings were rejected")
        try check(state.orderedAssets == outfitAssets && state.instruction == "Outfit-specific instruction" && state.settings.steps == 30,
                  "Parameter copying replaced template content")
        try check(state.settings.prompt.isEmpty && state.settings.assets.isEmpty && state.settings.initImageID == nil,
                  "Parameter copying retained Creation prompt or input bindings")
        let outfitDraft = try state.generationDraft()
        try check(outfitDraft.assets == outfitAssets && outfitDraft.initImageID == outfitAssets.first?.id &&
                  outfitDraft.prompt.contains("<image1> as the person reference") &&
                  outfitDraft.prompt.contains("<image2> as the clothing reference") &&
                  outfitDraft.prompt.contains("Outfit-specific instruction"),
                  "Outfit generation lost its ordered role references or instruction")

        state.selectTemplate(.identity)
        try check(state.settings.steps == 25 && state.orderedAssets.isEmpty, "Outfit changes leaked into the identity template")
        state.setInstruction("Identity-specific instruction")
        // Insert the optional role first; numbering must follow roles, not insertion order.
        let sceneImported = await state.importFiles([scene], replacing: .scene)
        try check(sceneImported && state.imageNumber(for: .scene) == 2 && state.imageNumber(for: .person) == nil && state.generationBlocker != nil,
                  "An optional scene without a person was renumbered to image1 or enabled generation")
        let personImported = await state.importFiles([person])
        try check(sceneImported && personImported && state.imageNumber(for: .person) == 1 && state.imageNumber(for: .scene) == 2,
                  "Role insertion order changed image numbering or group-fill destination")
        let identityAssets = state.orderedAssets
        try check(identityAssets.map(\.name) == [person.lastPathComponent, scene.lastPathComponent], "Identity inputs are not person then scene")
        let identityDraft = try state.generationDraft()
        try check(identityDraft.assets == identityAssets && identityDraft.initImageID == identityAssets.first?.id &&
                  identityDraft.prompt.contains("<image2> as the scene"), "Identity generation disagreed with role ordering")
        let beforeSlot = try bytes(state.document)
        let inputDirectory = directory.appendingPathComponent("playground-inputs")
        let filesBeforeSlot = try files(in: inputDirectory)
        let multiSlot = await state.importFiles([person, clothing], replacing: .person)
        let invalidRole = await state.importFiles([clothing], replacing: .clothing)
        let fullGroup = await state.importFiles([person])
        try check(!multiSlot && !invalidRole && !fullGroup && (try bytes(state.document)) == beforeSlot,
                  "A slot accepted multiple images, an invalid role or an overfilled group")
        try check(try files(in: inputDirectory) == filesBeforeSlot, "Rejected slot/group import staged an input")
        copiedSettings.steps = 40
        try check(state.syncSettings(from: copiedSettings) && state.orderedAssets == identityAssets && state.instruction == "Identity-specific instruction",
                  "Identity parameter copying lost role bindings or instruction")
        state.save()
        let persisted = try bytes(state.document)
        let reopened = PlaygroundState(directory: directory, initialSettings: StudioDraft(), models: [model])
        try check(reopened.saved && reopened.storageError == nil && reopened.template == .identity &&
                  (try bytes(reopened.document)) == persisted, "Independent template drafts did not reopen intact")
        try check(reopened.settings.steps == 40 && reopened.document.templates[PlaygroundTemplate.outfit.rawValue]?.settings.steps == 30,
                  "Template parameters were flattened during persistence")
        return reopened
    }

    @MainActor private static func failedImportRollback(_ state: PlaygroundState, directory: URL, person: URL, root: URL) async throws {
        state.selectTemplate(.identity)
        state.remove(.person); state.remove(.scene)
        let invalid = root.appendingPathComponent("not-an-image.png")
        try Data("invalid PNG bytes".utf8).write(to: invalid)
        let before = try bytes(state.document)
        let inputDirectory = directory.appendingPathComponent("playground-inputs")
        let filesBefore = try files(in: inputDirectory)
        // Both slots are empty so destination validation succeeds. The valid first
        // item stages a copy before the second item's decoding fails. Outfit's
        // existing bindings must survive, as must all previously retained copies.
        let failed = await state.importFiles([person, invalid])
        try check(!failed && !state.importing && state.message?.contains(invalid.lastPathComponent) == true &&
                  (try bytes(state.document)) == before, "A failing second image changed the document or left it busy")
        try check(try files(in: inputDirectory) == filesBefore, "Failed group import leaked its first staged copy or deleted retained images")
        state.selectTemplate(.outfit)
        let beforeReplacement = try bytes(state.document)
        let failedReplacement = await state.importFiles([invalid], replacing: .person)
        try check(!failedReplacement && (try bytes(state.document)) == beforeReplacement,
                  "An invalid replacement discarded the previous person binding")
        try check(try files(in: inputDirectory) == filesBefore, "An invalid replacement changed retained input files")
    }

    @MainActor private static func resultOwnership(_ state: PlaygroundState, directory: URL,
        model: StudioModel, initial: StudioDraft, output: URL) async throws {
        state.selectTemplate(.outfit)
        let submitted = try state.generationDraft()
        let personBefore = state.asset(for: .person)!
        let clothingBefore = state.asset(for: .clothing)!
        state.selectTemplate(.identity)
        state.recordSubmission(submitted, seed: 812, template: .outfit)
        try check(state.current.lastSeed == nil && state.document.templates[PlaygroundTemplate.outfit.rawValue]?.lastSeed == 812,
                  "Submission recorded the seed on the selected template instead of the submitted template")
        let recorded = state.document.templates[PlaygroundTemplate.outfit.rawValue]!.settings
        try check(recorded.prompt.isEmpty && recorded.assets.isEmpty && recorded.initImageID == nil && recorded.steps == submitted.steps,
                  "Submission persistence retained generation content or lost its parameters")
        var request = NativeRequest(prompt: submitted.prompt, output: output.path)
        request.model = model.id; request.operation = "image.edit"; request.steps = submitted.steps; request.seed = 812
        var job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "complete",
                            completed: submitted.steps, total: submitted.steps, elapsed: 1)
        job.workflowID = PlaygroundTemplate.outfit.workflowID
        state.recordResult(job, template: .outfit)
        try check(state.current.lastJobID == nil && state.document.templates[PlaygroundTemplate.outfit.rawValue]?.lastJobID == job.id,
                  "Result recorded its ID on the selected template instead of the originating template")
        let beforeWrongTemplate = try bytes(state.document)
        let inputDirectory = directory.appendingPathComponent("playground-inputs")
        let beforeWrongFiles = try files(in: inputDirectory)
        let wrongTemplate = await state.useResultAsPerson(job)
        try check(!wrongTemplate && (try bytes(state.document)) == beforeWrongTemplate &&
                  (try files(in: inputDirectory)) == beforeWrongFiles, "A result from another template was imported")
        state.selectTemplate(.outfit)
        let outputBytes = try Data(contentsOf: output)
        let reused = await state.useResultAsPerson(job)
        guard let newPerson = state.asset(for: .person) else { throw NativeFailure(message: "Result reuse removed the person slot") }
        try check(reused && newPerson.id != personBefore.id && newPerson.path != output.path && state.asset(for: .clothing) == clothingBefore &&
                  state.imageNumber(for: .person) == 1 && state.imageNumber(for: .clothing) == 2,
                  "Result reuse changed the clothing binding or used the result without an immutable person copy")
        try check(try Data(contentsOf: URL(fileURLWithPath: newPerson.path)) == outputBytes &&
                  (try Data(contentsOf: output)) == outputBytes && FileManager.default.fileExists(atPath: personBefore.path),
                  "Result reuse changed its source image or deleted a historical input")
        let reopened = PlaygroundState(directory: directory, initialSettings: initial, models: [model])
        try check(reopened.current.lastSeed == 812 && reopened.current.lastJobID == job.id && reopened.asset(for: .person) == newPerson,
                  "Result provenance, actual seed or new person binding did not persist")
    }

    @MainActor private static func damagedFileProtection(root: URL, model: StudioModel,
        initial: StudioDraft, validDocument: PlaygroundDocument) throws {
        var incomplete = validDocument
        incomplete.templates.removeValue(forKey: PlaygroundTemplate.identity.rawValue)
        var illegalRole = validDocument
        illegalRole.templates[PlaygroundTemplate.outfit.rawValue]?.roleAssets[PlaygroundRole.scene.rawValue] =
            validDocument.templates[PlaygroundTemplate.outfit.rawValue]!.roleAssets[PlaygroundRole.person.rawValue]!
        let payloads: [(String, Data)] = [("corrupt", Data("{broken".utf8)),
                                         ("incomplete", try bytes(incomplete)), ("illegal-role", try bytes(illegalRole))]
        for (name, payload) in payloads {
            let directory = root.appendingPathComponent("protected-\(name)")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let file = directory.appendingPathComponent("playground.json")
            try payload.write(to: file)
            let state = PlaygroundState(directory: directory, initialSettings: initial, models: [model])
            try check(state.storageError != nil && !state.saved && state.generationBlocker != nil,
                      "\(name) persistence was accepted or enabled generation")
            state.setInstruction("This must not overwrite the unread file")
            state.selectTemplate(.identity)
            _ = state.syncSettings(from: initial)
            state.save() // Also cancels the pending debounced save.
            try check(!state.saved && state.storageError != nil && (try Data(contentsOf: file)) == payload,
                      "\(name) persistence was overwritten by a later edit/save")
        }
    }

    @MainActor private static func preprocessingAndPending(root: URL, model: StudioModel,
        initial: StudioDraft, clothing: URL) async throws {
        let directory = root.appendingPathComponent("processing-and-pending")
        let state = PlaygroundState(directory: directory, initialSettings: initial, models: [model])
        let person = root.appendingPathComponent("large-person.png")
        try writePNG(to: person, width: 2048, height: 1024, color: CGColor(red: 0.3, green: 0.5, blue: 0.7, alpha: 1))
        let sourceBytes = try Data(contentsOf: person)
        let imported = await state.importFiles([person, clothing])
        let original = state.asset(for: .person)!, clothingAsset = state.asset(for: .clothing)!
        let settingsBytes = try bytes(state.settings)
        let fitted = await state.prepare(.person, preset: .fit512)
        try check(imported && fitted && state.asset(for: .person)?.id == original.id &&
                  state.asset(for: .person)?.width == 512 && state.asset(for: .person)?.height == 256 &&
                  state.asset(for: .clothing) == clothingAsset && (try bytes(state.settings)) == settingsBytes,
                  "Playground resize changed role identity, other slots or output parameters")
        let automatic = await state.prepare(.person, preset: .automatic)
        try check(automatic && state.asset(for: .person)?.width == 1024 && state.asset(for: .person)?.height == 512,
                  "Playground processed a smaller derivative instead of the retained original")
        let restored = await state.prepare(.person, preset: .original)
        try check(restored && state.asset(for: .person)?.path == original.path &&
                  (try Data(contentsOf: URL(fileURLWithPath: original.path))) == sourceBytes &&
                  (try Data(contentsOf: person)) == sourceBytes, "Playground restoration lost or changed original pixels")

        var incompatible = initial
        incompatible.upscaleAfterGeneration = true
        try check(state.syncSettings(from: incompatible) && state.generationBlocker?.contains("超分") == true &&
                  state.settings.upscaleAfterGeneration, "Playground silently ignored automatic upscaling")
        incompatible = initial; incompatible.width = 768
        try check(state.syncSettings(from: incompatible) && state.generationBlocker?.contains("512×512") == true &&
                  state.settings.activeLoRAs == initial.activeLoRAs && state.settings.width == 768,
                  "Playground bypassed or normalized an incompatible ordinary LoRA canvas")
        incompatible = initial; incompatible.loras = []; incompatible.qwen21DiTCache = "balanced"; incompatible.steps = 6
        try check(state.syncSettings(from: incompatible) && state.generationBlocker?.contains("20–40") == true &&
                  state.settings.qwen21DiTCache == "balanced" && state.settings.steps == 6,
                  "Playground disabled the cache or silently changed an incompatible schedule")
        try check(state.syncSettings(from: initial), "Cannot restore compatible settings")

        // Hold an actual NSItemProvider callback so guards are exercised after
        // importing is set, rather than in @Published's willSet notification.
        let deferred = DeferredPlaygroundImageProvider(), provider = deferred.makeProvider()
        let beforePending = try bytes(state.document)
        let pending = Task { await state.importProviders([provider], replacing: .person) }
        try await deferred.waitUntilRequested()
        state.selectTemplate(.identity); state.remove(.clothing); state.setInstruction("Must stay locked")
        let syncDuringImport = state.syncSettings(from: incompatible)
        let prepareDuringImport = await state.prepare(.person, preset: .fit512)
        try check(state.importing && !syncDuringImport && !prepareDuringImport &&
                  (try bytes(state.document)) == beforePending && state.generationBlocker != nil,
                  "A pending provider allowed template/role/instruction/parameter mutation or generation")
        deferred.finish(sourceBytes)
        let completed = await pending.value
        try check(completed && !state.importing && state.asset(for: .clothing) == clothingAsset &&
                  state.asset(for: .person)?.id != original.id && FileManager.default.fileExists(atPath: original.path),
                  "Provider replacement changed the other role or deleted a historical input")

        let inputs = directory.appendingPathComponent("playground-inputs")
        let beforeCancel = try bytes(state.document), filesBeforeCancel = try files(in: inputs)
        let cancelledProvider = DeferredPlaygroundImageProvider(), cancelledItem = cancelledProvider.makeProvider()
        let cancelled = Task { await state.importProviders([cancelledItem], replacing: .person) }
        try await cancelledProvider.waitUntilRequested()
        cancelled.cancel(); cancelledProvider.finish(sourceBytes)
        let cancelledResult = await cancelled.value
        try check(!cancelledResult && !state.importing && (try bytes(state.document)) == beforeCancel &&
                  (try files(in: inputs)) == filesBeforeCancel, "Cancelled provider import changed its slot or leaked an input")

        let staleProvider = DeferredPlaygroundImageProvider(), staleItem = staleProvider.makeProvider()
        let stale = Task { await state.importProviders([staleItem], replacing: .person) }
        try await staleProvider.waitUntilRequested()
        // Cross-template bookkeeping may arrive independently of the editor.
        state.recordSubmission(initial, seed: 731, template: .identity)
        let newerDocument = try bytes(state.document)
        staleProvider.finish(sourceBytes)
        let staleResult = await stale.value
        try check(!staleResult && !state.importing && (try bytes(state.document)) == newerDocument &&
                  (try files(in: inputs)) == filesBeforeCancel,
                  "A stale provider callback replaced a newer context or leaked its staged copy")
    }

    private static func modelFixture() throws -> StudioModel {
        try JSONDecoder().decode(StudioModel.self, from: Data(#"{"id":"qwen-image-2.1","name":"CPU descriptor fixture","executor":true,"output":"image","operations":["image.generate","image.edit"],"default_steps":40,"default_frames":1,"default_width":512,"default_height":512,"inputs":["image","text"],"max_images":10,"supports_lora":true,"lora_strategies":["inference_time"]}"#.utf8))
    }
    private static func bytes<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }
    private static func files(in directory: URL) throws -> Set<String> {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        return Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
    }
    private static func writePNG(to url: URL, width: Int, height: Int, color: CGColor) throws {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                  bytesPerRow: width * 4, space: space,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
            throw NativeFailure(message: "Cannot allocate small PNG fixture")
        }
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        guard let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw NativeFailure(message: "Cannot create small PNG fixture")
        }
        CGImageDestinationAddImage(destination, image, nil)
        try check(CGImageDestinationFinalize(destination), "Cannot save small PNG fixture")
    }
}
