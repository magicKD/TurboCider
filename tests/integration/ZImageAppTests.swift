import AppKit
import Foundation

@main
struct ZImageAppTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 7 else { throw NativeFailure(message: "z-image-app-tests MODEL SHARED_TEXT LORA BASE_MANIFEST LORA_MANIFEST OUTPUT") }
        let root = URL(fileURLWithPath: args[6])
        let studio = StudioState(directory: root)
        let store = NativeJobStore(directory: root)
        func check(_ value: @autoclosure () -> Bool, _ message: String) throws {
            guard value() else { throw NativeFailure(message: message) }
        }
        try studio.installZImage(model: URL(fileURLWithPath: args[1]), sharedText: URL(fileURLWithPath: args[2]))
        studio.draft.prompt = "A cinematic red fox walking through fresh snow, soft morning light."
        studio.draft.acceleration = StudioAcceleration(policy: "gpu")
        studio.draft.randomSeed = false; studio.draft.seedText = "42"
        let model = URL(fileURLWithPath: studio.draft.modelPath)
        var results: [String: Any] = [:]
        let configurations = root.appendingPathComponent("configurations")
        try FileManager.default.createDirectory(at: configurations, withIntermediateDirectories: true)
        func saveConfiguration(_ name: String) throws {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(studio.draft).write(to: configurations.appendingPathComponent(name + ".json"), options: .atomic)
        }
        func generate(_ name: String) async throws {
            let request = try studio.draft.request(output: root.appendingPathComponent(name + ".png"))
            let job = try await store.generate(modelURL: model, request: request)
            try check(job.state == "succeeded", "App job failed: \(name)")
            guard let image = NSBitmapImageRep(data: try Data(contentsOf: URL(fileURLWithPath: request.output))) else { throw NativeFailure(message: "App output cannot decode") }
            try check(image.pixelsWide == 512 && image.pixelsHigh == 512, "Wrong App output shape")
            results[name] = try JSONSerialization.jsonObject(with: Data(job.resultJSON!.utf8))
            print("PASS generated \(name)", terminator: "\n")
        }
        try saveConfiguration("01-z-image-gpu")
        try await generate("base-gpu")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: args[4])
        try saveConfiguration("02-z-image-gpu-ane")
        try await generate("base-gpu-ane")
        studio.draft.loras = [StudioLoRA(path: args[3], strength: 1.0)]
        let guarded = try studio.draft.request(output: root.appendingPathComponent("unused.png"))
        try check(guarded.execution == "gpu", "Base ANE artifact accepted active LoRA")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu")
        try saveConfiguration("03-z-image-lora-gpu")
        let preparation = try studio.preparationRequest(modelID: "z-image-turbo", output: root.appendingPathComponent("unused-prepare.png"))
        try check(preparation.loras?.count == 1 && preparation.width == 512 && preparation.steps == 9, "Load discarded LoRA or task settings")
        try await store.prepare(modelURL: model, request: preparation, warmup: false)
        let prepared = try JSONSerialization.jsonObject(with: Data(store.resourceReport!.utf8)) as! [String: Any]
        try check(prepared["prepared"] as? Bool == true && !FileManager.default.fileExists(atPath: preparation.output), "Load unexpectedly generated an image")
        try await generate("lora-gpu")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: args[5])
        try saveConfiguration("04-z-image-lora-gpu-ane")
        try await generate("lora-gpu-ane")
        for name in ["lora-gpu", "lora-gpu-ane"] {
            let result = results[name] as! [String: Any]
            try check(result["lora_applied_projections"] as? Int == 180, "Requested adapter was not fully applied")
        }
        let hybrid = (results["lora-gpu-ane"] as! [String: Any])["hybrid"] as! [String: Any]
        try check(hybrid["lora_identity_verified"] as? Bool == true, "LoRA provenance not verified")
        studio.save()
        let restored = StudioState(directory: root)
        try check(restored.draft.modelPath == model.path && restored.draft.loras.count == 1 && restored.draft.acceleration?.manifest == args[5], "App draft did not persist model/LoRA/ANE")
        let history = NativeJobStore(directory: root)
        try check(history.jobs.filter { $0.state == "succeeded" }.count >= 4, "App history did not persist generations")
        try await store.unload()
        let report: [String: Any] = ["passed": true, "checks": ["comfy_shared_text_installation", "base_gpu", "base_gpu_ane", "lora_preserved_on_load", "load_without_generation", "lora_gpu", "lora_gpu_ane", "base_manifest_rejected_for_lora", "draft_and_history_persistence", "unload"], "results": results]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("report.json"), options: .atomic)
        print("PASS Z-Image App end-to-end")
    }
}
