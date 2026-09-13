import Foundation

@main struct ZImageShapeTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 4 else { throw NativeFailure(message: "z-image-shape-tests MODEL EXPERIMENT_ROOT OUTPUT") }
        let model = args[1], root = URL(fileURLWithPath: args[2])
        func check(_ ok: Bool, _ message: String) throws { if !ok { throw NativeFailure(message: message) } }
        func manifest(_ mode: String) throws -> String {
            let value = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("compile-\(mode).json"))) as! [String: Any]
            return value["manifest"] as! String
        }
        let short = "A cinematic red fox walking through fresh snow, soft morning light."
        let long = short + String(repeating: " snow", count: 491)
        try check(try NativeEngine.zImageTokenCount(modelPath: model, prompt: short) == 21, "Short count mismatch")
        try check(try NativeEngine.zImageTokenCount(modelPath: model, prompt: long) == 512, "512-token boundary mismatch")
        try check(try NativeEngine.zImageTokenCount(modelPath: model, prompt: long + " snow") == 513, "Extended prompt count mismatch")
        try check(try NativeEngine.zImageTokenCount(modelPath: model, prompt: long + String(repeating: " snow", count: 513)) == 1025, "Overlimit UI count must remain available")
        let flexible = try manifest("enumerated"), fixed = try manifest("fixed")
        let oldData = try Data(contentsOf: root.deletingLastPathComponent().appendingPathComponent("z-image-m4pro-512-20260907/ane-a8192-b1056.compile.json"))
        let old = (try JSONSerialization.jsonObject(with: oldData) as! [String: Any])["manifest"] as! String
        for (required, expected) in [(1056,1056),(1088,1088),(1536,1536)] {
            try check(AccelerationDiscovery.find(modelPath: model, preferred: flexible, minimumRows: required, modelID: "z-image-turbo")?.rows == expected, "Flexible cache selection failed")
        }
        try check(AccelerationDiscovery.find(modelPath: model, preferred: flexible, minimumRows: 1568, modelID: "z-image-turbo") == nil, "Oversized request accepted")
        let store = NativeJobStore(directory: URL(fileURLWithPath: args[3]))
        var draft = StudioDraft()
        draft.modelID = "z-image-turbo"; draft.modelPaths["z-image-turbo"] = model
        draft.width = 512; draft.height = 512; draft.steps = 9; draft.prompt = long
        draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: old)
        draft.acceleration?.knownManifests = [flexible, fixed]
        let resolved = try await store.resolveAcceleration(draft)
        try check(resolved.acceleration?.manifest != old, "Long prompt reused undersized warm cache")
        try check(store.accelerationStatus?.contains("未重新编译") == true, "Existing cache not reused")
        draft.acceleration?.knownManifests = []
        draft.acceleration?.coreMLCache = URL(fileURLWithPath: args[3]).appendingPathComponent("empty-cache").path
        do {
            _ = try await store.resolveAcceleration(draft)
            throw NativeFailure(message: "Old 32-token cache accepted a 512-token prompt")
        } catch { try check(error.localizedDescription.contains("1536"), "Capacity error does not explain required rows") }
        print("PASS: exact token count, extended and overlimit counting, variable cache routing, legacy capacity rejection, no recompilation")
    }
}
