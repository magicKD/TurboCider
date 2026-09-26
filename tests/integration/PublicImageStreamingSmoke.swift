import Foundation

/// Opt-in real-model App/worker smoke. It uses native discovery, never a mocked
/// provider. A test-catalog build proves integration, not release qualification.
@main struct PublicImageStreamingSmoke {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 3 else {
            throw NativeFailure(message: "Usage: public-streaming-smoke NEW_OUTPUT_DIRECTORY LOCAL_Z_IMAGE_MODEL")
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let model = URL(fileURLWithPath: CommandLine.arguments[2])
        guard !FileManager.default.fileExists(atPath: root.path) else {
            throw NativeFailure(message: "Output directory already exists")
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        func check(_ condition: @autoclosure () -> Bool, _ reason: String) throws {
            guard condition() else { throw NativeFailure(message: reason) }
        }
        let studioDirectory = root.appendingPathComponent("studio")
        let studio = StudioState(directory: studioDirectory)
        studio.draft.modelID = "z-image-turbo"
        studio.draft.modelPaths["z-image-turbo"] = model.path
        studio.draft.prompt = "A red fox standing in fresh snow, pine forest, soft morning sunlight, detailed photograph."
        studio.draft.width = 512; studio.draft.height = 512; studio.draft.steps = 9
        studio.draft.seedText = "42"; studio.draft.randomSeed = false
        studio.draft.acceleration = StudioAcceleration(policy: "gpu")
        await studio.refreshStreamingOptions()
        if let options = studio.streamingOptions {
            try JSONEncoder().encode(options).write(to: root.appendingPathComponent("options.json"))
        }
        try check(studio.streamingOptionsError == nil, "Native discovery failed: \(studio.streamingOptionsError ?? "")")
        try check(studio.streamingOption(for: .tier16)?.status == "available", "16 GiB target not available through real native discovery")
        studio.setStreamingSelection(.tier16)
        studio.save()
        let restoredStudio = StudioState(directory: studioDirectory)
        try check(restoredStudio.draft.streaming.selection == .tier16 && restoredStudio.draft.prompt == studio.draft.prompt,
                  "Streaming selection or prompt was not persisted")
        var observations: [[String: Any]] = []
        // Success, mid-denoise cancellation, then success prove worker cleanup
        // and that cancellation does not poison the next App request.
        let directory = root.appendingPathComponent("jobs")
        let store = NativeJobStore(directory: directory)
        for (index, cancel) in [false, true, false].enumerated() {
            let output = directory.appendingPathComponent("outputs/image-\(index).png")
            let request = try restoredStudio.draft.publicStreamingRequest(output: output)
            try check(request.v2 != nil, "Public selection did not create a V2 request")
            let task = Task { try await store.generate(modelURL: model, request: request.legacy, streamingRequest: request.v2) }
            var sawDenoise = false
            if cancel {
                let deadline = Date().addingTimeInterval(180)
                while Date() < deadline {
                    if let job = store.activeJob, job.phase == "denoise", job.completed >= 1 {
                        sawDenoise = true; break
                    }
                    if let job = store.jobs.first, job.request.output == output.path, job.isTerminal { break }
                    try await Task.sleep(for: .milliseconds(25))
                }
                store.cancel()
            }
            var failure: Error?
            do { _ = try await task.value } catch { failure = error }
            guard let job = store.jobs.first else { throw NativeFailure(message: "No persisted job") }
            try JSONEncoder().encode(job).write(to: root.appendingPathComponent("job-\(index).json"))
            try check(!store.busy && !store.workerCleanupPending && !store.canUnload, "Worker not fully released")
            try check(job.publicWorker?.exitConfirmed == true, "Worker exit not confirmed")
            try check(job.publicWorker?.runtimeFingerprint == NativeEngine.runtimeBuildIdentity(), "Worker build identity differs")
            try check(job.publicStreamingTargetBytes == 16 << 30 && job.publicStreamingIntentJSON != nil, "Durable intent missing")
            if cancel {
                try check(sawDenoise, "Cancellation did not reach denoising")
                try check(failure is CancellationError && job.state == "cancelled", "Cancellation failed")
                try check(!FileManager.default.fileExists(atPath: output.path), "Cancelled request published an image")
            } else {
                try check(failure == nil && job.state == "succeeded", "Generation failed: \(failure?.localizedDescription ?? job.state)")
                try check(job.publicStreamingResolutionJSON != nil && job.resultJSON != nil, "Verified result/receipt missing")
                try check(FileManager.default.fileExists(atPath: output.path), "Published image missing")
            }
            let entries = try FileManager.default.contentsOfDirectory(atPath: directory.appendingPathComponent("outputs").path)
            try check(!entries.contains { $0.hasPrefix(".tc-image-staging-") }, "Staged output leaked")
            let restored = NativeJobStore(directory: directory)
            try check(!restored.busy && restored.jobs.first?.id == job.id && restored.jobs.first?.state == job.state,
                      "Restart did not preserve terminal job")
            observations.append(["index": index, "state": job.state, "worker_exit_confirmed": true,
                                 "elapsed": job.elapsed, "runtime_identity": NativeEngine.runtimeBuildIdentity()])
            print("App streaming \(index): \(job.state), worker exit and durable intent verified")
        }
        let first = try Data(contentsOf: directory.appendingPathComponent("outputs/image-0.png"))
        let last = try Data(contentsOf: directory.appendingPathComponent("outputs/image-2.png"))
        try check(first == last, "Same seed changed after cancellation/restart")
        try JSONSerialization.data(withJSONObject: ["passed": true, "observations": observations,
            "same_seed_png_identical": true, "qualification": "integration only; consult bundled catalog provenance"],
            options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("report.json"))
    }
}
