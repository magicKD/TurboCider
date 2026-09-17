import Foundation

/// Real tokenizer and registered manifests; no inference or artifact mutation.
@main struct ZImagePartitionRoutingTests {
    @MainActor static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 5 else {
            throw NativeFailure(message: "z-image-partition-routing-tests MODEL SMALL_MANIFEST LARGE_MANIFEST OUTPUT")
        }
        guard AccelerationDiscovery.optimizationEnabled("z_image_smallest_partition") else {
            throw NativeFailure(message: "This routing probe requires the measured M5 Pro 24 GiB device policy")
        }
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw NativeFailure(message: message) }
        }
        let output = URL(fileURLWithPath: args[4])
        let store = NativeJobStore(directory: output)
        var draft = StudioDraft()
        draft.modelID = "z-image-turbo"
        draft.modelPaths[draft.modelID] = args[1]
        draft.steps = 8; draft.residency = "streamed"; draft.zImageStreamingBudgetGiB = 10
        draft.prompt = "A cinematic red fox walking through fresh snow, soft morning light."
        let tokens = try NativeEngine.zImageTokenCount(modelPath: args[1], prompt: draft.prompt)
        draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: args[3])
        draft.acceleration?.knownManifests = [args[2], args[3]]
        var observations: [[String: Any]] = []
        for size in [512, 1024, 512] {
            draft.width = size; draft.height = size
            draft = try await store.resolveAcceleration(draft)
            let expected = size == 512 ? args[2] : args[3]
            let request = try draft.request(output: output.appendingPathComponent("unused.png"))
            try check(request.ane_manifest == expected, "Wrong partition after resizing to \(size)")
            try check(request.execution == "gpu_ane" && request.residency == "streamed",
                      "Resolution change lost the hybrid streaming route")
            try check(store.accelerationStatus?.contains("未重新编译") == true,
                      "Existing compiled partition was not reused")
            observations.append(["size": size, "text_tokens": tokens,
                                 "manifest": request.ane_manifest ?? "",
                                 "status": store.accelerationStatus ?? ""])
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: observations, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("routing.json"))
        print("PASS: oversized preference → 512 → 1024 → 512 selects matching registered partitions without recompilation")
    }
}
