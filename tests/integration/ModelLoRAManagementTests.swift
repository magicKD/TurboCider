import Foundation

/// CPU-only state regression. Tiny presence fixtures are deliberately not
/// adapters: no catalog, request/plan, tensor loader or inference is invoked.
@main struct ModelLoRAManagementTests {
    private static func check(_ value: @autoclosure () -> Bool, _ reason: String) throws {
        guard value() else { throw NativeFailure(message: reason) }
    }

    private struct Fixtures {
        let root: URL
        let models: [StudioModel]
        let turbo: URL
        let ordinary: URL

        init(root: URL) throws {
            self.root = root
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            models = try JSONDecoder().decode([StudioModel].self, from: Data(#"""
            [
                {"id":"qwen-image-2.1","name":"Qwen state fixture","executor":true,"output":"image","operations":["image.generate","image.edit"],"default_steps":40,"default_frames":1,"default_width":512,"default_height":512,"inputs":["text","image"],"max_images":10,"supports_lora":true,"supports_gpu_ane":true,"lora_strategies":["auto","inference_time"]},
                {"id":"flux2-klein-4b","name":"FLUX state fixture","executor":true,"output":"image","operations":["image.generate","image.edit"],"default_steps":4,"default_frames":1,"default_width":512,"default_height":512,"inputs":["text","image"],"supports_lora":true,"lora_strategies":["auto","inference_time"]}
            ]
            """#.utf8))
            turbo = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
            ordinary = root.appendingPathComponent("ordinary.safetensors")
            for url in [turbo, ordinary] {
                try Data("State-only presence fixture; not model weights.".utf8).write(to: url)
            }
        }

        @MainActor func state(_ name: String) throws -> StudioState {
            let directory = root.appendingPathComponent(name)
            let file = directory.appendingPathComponent("studio-draft.json")
            if !FileManager.default.fileExists(atPath: file.path) {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                var initial = StudioDraft()
                // Avoid consulting a user's remembered model path on first init.
                initial.modelPaths = ["qwen-image-2.1": root.path, "flux2-klein-4b": root.path]
                initial.acceleration = StudioAcceleration(policy: "gpu")
                try JSONEncoder().encode(initial).write(to: file, options: .atomic)
            }
            return StudioState(directory: directory, models: models)
        }
    }

    @MainActor static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-lora-management-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let fixtures = try Fixtures(root: root)
        try turboActivationAndEnableCycle(fixtures)
        try modelAssociation(fixtures)
        try normalizedPathsAndLimit(fixtures)
        try ordinaryStrengthPersistence(fixtures)
        print("PASS ModelLoRAManagement: Viggle preset and enable cycle, per-model restoration, normalized deduplication/8-item limit, ordinary strengths and persistence (state only; no adapters loaded)")
    }

    @MainActor private static func assertTurbo(_ studio: StudioState, id: UUID) throws {
        try check(studio.draft.modelID == "qwen-image-2.1" && studio.draft.qwen21TurboLoRA?.id == id,
                  "Supported Viggle adapter lost its Qwen-only identity")
        try check(studio.draft.width == 512 && studio.draft.height == 512 && studio.draft.steps == 6,
                  "Viggle activation did not apply 512-square/six-step sampling")
        try check(studio.draft.acceleration?.policy == "gpu" && studio.draft.profilePath.isEmpty &&
                  studio.draft.qwen21DiTCache == "off" && studio.draft.loraStrategy == "inference_time",
                  "Viggle activation retained an incompatible device, cache or loading strategy")
        try check(studio.draft.qwen21TurboLoRA?.strength == 1 && studio.draft.qwen21TurboLoRA?.role == "transformer" &&
                  studio.draft.qwen21TurboConfigurationIssues.isEmpty,
                  "Viggle activation did not preserve strength-one Transformer contract")
    }

    @MainActor private static func turboActivationAndEnableCycle(_ fixtures: Fixtures) throws {
        let studio = try fixtures.state("turbo")
        studio.selectModel("qwen-image-2.1")
        studio.draft.steps = 40; studio.draft.qwen21DiTCache = "balanced"
        studio.draft.width = 768; studio.draft.height = 512
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
        studio.draft.profilePath = "/state-only/old-profile.json"
        studio.draft.prompt = "Retain this instruction across LoRA management."
        studio.draft.seedText = "42"
        studio.addLoRA(fixtures.turbo.path)
        try check(studio.draft.loras.count == 1, "Adding one Viggle adapter created the wrong number of bindings")
        let id = studio.draft.loras[0].id
        try assertTurbo(studio, id: id)
        try check(studio.saved && studio.draft.prompt == "Retain this instruction across LoRA management." && studio.draft.seedText == "42",
                  "Viggle preset lost prompt/seed or was not saved")

        studio.setLoRAEnabled(id, enabled: false)
        try check(studio.draft.loras.count == 1 && studio.draft.loras[0].id == id && !studio.draft.loras[0].enabled &&
                  studio.draft.activeLoRAs.isEmpty && studio.draft.qwen21TurboLoRA == nil && studio.draft.steps == 40,
                  "Disabling Viggle removed its binding or left the base model on six steps")
        let reopened = try fixtures.state("turbo")
        try check(reopened.draft.loras == studio.draft.loras && reopened.draft.steps == 40,
                  "Disabled Viggle binding did not survive a draft reload")
        reopened.draft.steps = 25; reopened.draft.qwen21DiTCache = "balanced"
        reopened.setLoRAEnabled(id, enabled: true)
        try assertTurbo(reopened, id: id)
        let finalReload = try fixtures.state("turbo")
        try assertTurbo(finalReload, id: id)
        print("PASS: add/disable/reload/re-enable applies the same Viggle contract")
    }

    @MainActor private static func modelAssociation(_ fixtures: Fixtures) throws {
        let studio = try fixtures.state("association")
        studio.selectModel("qwen-image-2.1"); studio.addLoRA(fixtures.turbo.path)
        let qwenBindings = studio.draft.loras
        studio.selectModel("flux2-klein-4b")
        try check(studio.draft.loras.isEmpty && studio.draft.qwen21TurboLoRA == nil && studio.draft.steps == 4,
                  "Switching to FLUX leaked Qwen's adapter or schedule")
        studio.addLoRA(fixtures.ordinary.path)
        studio.draft.loras[0].strength = 0.625
        let fluxID = studio.draft.loras[0].id
        studio.setLoRAEnabled(fluxID, enabled: false)
        let fluxBindings = studio.draft.loras
        studio.selectModel("qwen-image-2.1")
        try check(studio.draft.loras == qwenBindings, "Returning to Qwen lost its per-model adapter selection")
        try assertTurbo(studio, id: qwenBindings[0].id)
        studio.selectModel("flux2-klein-4b")
        try check(studio.draft.loras == fluxBindings && studio.draft.steps == 4 && studio.draft.qwen21TurboLoRA == nil,
                  "Returning to FLUX lost its strength/disabled state or acquired Turbo sampling")
        studio.save()
        let reopened = try fixtures.state("association")
        try check(reopened.draft.loras == fluxBindings, "Active model's LoRA settings did not persist")
        reopened.selectModel("qwen-image-2.1")
        try check(reopened.draft.loras == qwenBindings, "Inactive model's LoRA settings did not persist")
        reopened.selectModel("flux2-klein-4b")
        try check(reopened.draft.loras == fluxBindings, "Model switching after reload lost the other model's LoRA settings")
        reopened.save()
        print("PASS: model switching/reload preserves independent bindings, IDs, strengths and enabled flags")
    }

    @MainActor private static func normalizedPathsAndLimit(_ fixtures: Fixtures) throws {
        let studio = try fixtures.state("limit")
        studio.selectModel("flux2-klein-4b"); studio.addLoRA(fixtures.ordinary.path)
        studio.draft.loras[0].strength = 0.75
        let id = studio.draft.loras[0].id
        studio.setLoRAEnabled(id, enabled: false)
        let subdirectory = fixtures.root.appendingPathComponent("path-alias")
        try FileManager.default.createDirectory(at: subdirectory, withIntermediateDirectories: true)
        let equivalent = fixtures.root.path + "/./path-alias/../" + fixtures.ordinary.lastPathComponent
        studio.addLoRA(equivalent)
        try check(studio.draft.loras.count == 1 && studio.draft.loras[0].id == id &&
                  studio.draft.loras[0].path == fixtures.ordinary.standardizedFileURL.path &&
                  studio.draft.loras[0].strength == 0.75 && !studio.draft.loras[0].enabled,
                  "Equivalent normalized path duplicated or reset an existing adapter")
        for index in 1...7 {
            let file = fixtures.root.appendingPathComponent("ordinary-\(index).safetensors")
            try Data([UInt8(index)]).write(to: file)
            studio.addLoRA(file.path)
        }
        try check(studio.draft.loras.count == 8 && studio.draft.activeLoRAs.count == 7,
                  "LoRA limit did not count disabled configured bindings")
        let before = studio.draft.loras
        let ninth = fixtures.root.appendingPathComponent("ninth.safetensors")
        try Data([9]).write(to: ninth)
        studio.addLoRA(ninth.path)
        try check(studio.draft.loras == before && studio.message?.contains("8") == true,
                  "A ninth binding bypassed the limit or failed without an actionable count")
        studio.addLoRA(equivalent)
        try check(studio.draft.loras == before, "Duplicate insertion at the limit changed existing bindings")
        studio.save()
        print("PASS: normalized paths deduplicate and the eight-binding cap includes disabled adapters")
    }

    @MainActor private static func ordinaryStrengthPersistence(_ fixtures: Fixtures) throws {
        let studio = try fixtures.state("ordinary")
        studio.selectModel("qwen-image-2.1")
        studio.draft.steps = 25; studio.draft.qwen21DiTCache = "balanced"
        studio.addLoRA(fixtures.ordinary.path)
        let ordinaryID = studio.draft.loras[0].id
        try check(studio.draft.loras[0].strength == 1 && studio.draft.steps == 25 &&
                  studio.draft.qwen21DiTCache == "balanced" && studio.draft.qwen21TurboLoRA == nil,
                  "Adding an ordinary LoRA changed the user's base sampling/cache settings")
        studio.draft.loras[0].strength = -0.375
        studio.setLoRAEnabled(ordinaryID, enabled: false)
        studio.addLoRA(fixtures.turbo.path)
        let turboID = studio.draft.loras.first { $0.path == fixtures.turbo.path }!.id
        try assertTurbo(studio, id: turboID)
        try check(studio.draft.loras.first { $0.id == ordinaryID }?.strength == -0.375 &&
                  studio.draft.loras.first { $0.id == ordinaryID }?.enabled == false,
                  "Turbo activation overwrote the inactive ordinary adapter's user strength")
        studio.removeLoRA(turboID)
        studio.setLoRAEnabled(ordinaryID, enabled: true)
        studio.draft.steps = 25; studio.draft.qwen21DiTCache = "balanced"
        for strength in [-8.0, -0.375, 0.625, 8.0] {
            studio.draft.loras[0].strength = strength
            studio.save()
            let reopened = try fixtures.state("ordinary")
            try check(reopened.draft.loras.count == 1 && reopened.draft.loras[0].id == ordinaryID &&
                      reopened.draft.loras[0].enabled && reopened.draft.loras[0].strength == strength &&
                      reopened.draft.steps == 25 && reopened.draft.qwen21DiTCache == "balanced" &&
                      reopened.draft.qwen21TurboLoRA == nil,
                      "Ordinary LoRA user strength/base settings failed persistence after Turbo removal: \(strength)")
        }
        print("PASS: ordinary LoRA strength survives Turbo transitions and persists across the supported range")
    }
}
