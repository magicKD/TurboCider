import AppKit
import Foundation

@main
struct StudioControlsTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 4 else { throw NativeFailure(message: "studio-controls-tests BASE_CONFIG LORA_CONFIG OUTPUT") }
        let root = URL(fileURLWithPath: args[3])
        let base = try JSONDecoder().decode(StudioDraft.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
        let lora = try JSONDecoder().decode(StudioDraft.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
        let store = NativeJobStore(directory: root)
        let studio = StudioState(directory: root)
        studio.draft = lora
        let manifests = [base.acceleration!.manifest, lora.acceleration!.manifest]
        studio.draft.acceleration?.knownManifests = manifests
        func check(_ value: Bool, _ message: String) throws {
            guard value else { throw NativeFailure(message: message) }
        }
        var results: [String: Any] = [:]
        func generate(_ name: String) async throws {
            let resolved = try await store.resolveAcceleration(studio.draft)
            studio.rememberAcceleration(resolved)
            let output = root.appendingPathComponent(name + ".png")
            let request = try resolved.request(output: output)
            let job = try await store.generate(modelURL: URL(fileURLWithPath: resolved.modelPath), request: request)
            results[name] = try JSONSerialization.jsonObject(with: Data(job.resultJSON!.utf8))
        }
        studio.draft.loras[0].enabled = false
        try await generate("disabled-lora-ane")
        try check(studio.draft.acceleration?.manifest == manifests[0], "Disabling LoRA did not select base cache")
        try check(store.accelerationStatus?.contains("未重新编译") == true, "Base cache was not reused")
        studio.draft.loras[0].enabled = true
        studio.draft.loras[0].strength = 0.5
        do {
            _ = try await store.resolveAcceleration(studio.draft)
            throw NativeFailure(message: "Mismatched LoRA strength accepted by ANE")
        } catch { try check(error.localizedDescription.contains("没有匹配"), "Wrong mismatch error: \(error)") }
        studio.setANEEnabled(false)
        try await generate("lora-half-gpu")
        studio.draft.loras[0].strength = 1.0
        studio.setANEEnabled(true)
        try await generate("lora-one-ane")
        try check(studio.draft.acceleration?.manifest == manifests[1], "Enabling LoRA did not restore its cache")
        try check(store.accelerationStatus?.contains("未重新编译") == true, "LoRA cache was not reused")
        let half = results["lora-half-gpu"] as! [String: Any]
        let one = results["lora-one-ane"] as! [String: Any]
        try check(half["lora_applied_projections"] as? Int == 180 && one["lora_applied_projections"] as? Int == 180, "LoRA was not fully applied")
        try check((one["hybrid"] as? [String: Any])?["lora_identity_verified"] as? Bool == true, "LoRA cache provenance was not verified")
        studio.setANEEnabled(false)
        studio.save()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(studio.draft).write(to: root.appendingPathComponent("app-controls-config.json"))
        try await store.unload()
        try JSONSerialization.data(withJSONObject: ["passed": true, "results": results], options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("report.json"))
        print("PASS: LoRA off/on, adjustable strength, GPU default, base/LoRA ANE cache reuse and mismatch rejection")
    }
}
