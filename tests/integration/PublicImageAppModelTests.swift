import Foundation

/// Actual App job path with local models. Empty public catalog means rejection;
/// this is lifecycle/integration evidence, not a successful generation campaign.
@main struct PublicImageAppModelTests {
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let executable = URL(fileURLWithPath: CommandLine.arguments[2])
        guard !FileManager.default.fileExists(atPath: root.path) else { throw NativeFailure(message: "Experiment directory already exists") }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for (model, path) in [("z-image-turbo", "/Users/chencanhui/models/TurboCider/Z-Image-Turbo"),
                              ("flux2-klein-4b", "/Users/chencanhui/models/TurboCider/FLUX.2-klein-4B")] {
            for cancel in [false, true] {
                let directory = root.appendingPathComponent(model + (cancel ? "-cancel" : "-reject"))
                let store = NativeJobStore(directory: directory, workerExecutable: executable)
                var request = NativeRequest(prompt: "A red fox in a snowy forest", output: directory.appendingPathComponent("outputs/output.png").path)
                request.model = model; request.width = 512; request.height = 512; request.steps = model == "z-image-turbo" ? 9 : 4
                let intent = NativeRequestV2(legacy: request, targetBytes: 10 << 30)
                let plan: [String: Any] = ["scope": "Real model through App job/worker path; no public qualification", "model": model,
                    "executable": executable.path, "runtime_identity": NativeEngine.runtimeBuildIdentity(), "cancel": cancel,
                    "expected": cancel ? "cancelled after starting worker; exit confirmed" : "failed: catalog_has_no_public_records; exit confirmed"]
                try JSONSerialization.data(withJSONObject: plan, options: .prettyPrinted).write(to: directory.appendingPathComponent("plan.json"))
                let task = Task { try await store.generate(modelURL: URL(fileURLWithPath: path), request: request, streamingRequest: intent) }
                if cancel {
                    for _ in 0..<200 {
                        if let job = store.jobs.first, let reference = job.publicWorker,
                           reference.admission(jobID: job.id, store: directory).observe() == .present { break }
                        try await Task.sleep(for: .milliseconds(25))
                    }
                    store.cancel()
                }
                var failure: Error?
                do { _ = try await task.value } catch { failure = error }
                guard let job = store.jobs.first else { throw NativeFailure(message: "Job missing") }
                let state: [String: Any] = ["busy": store.busy, "cleanup_pending": store.workerCleanupPending,
                    "state": job.state, "worker_exit_confirmed": job.publicWorker?.exitConfirmed ?? false,
                    "has_reusable_engine": store.canUnload, "error": failure?.localizedDescription ?? ""]
                try JSONSerialization.data(withJSONObject: state, options: .prettyPrinted).write(to: directory.appendingPathComponent("observation.json"))
                precondition(!store.busy && !store.workerCleanupPending && !store.canUnload && job.publicWorker?.exitConfirmed == true)
                precondition(job.state == (cancel ? "cancelled" : "failed") && job.publicStreamingResolutionJSON == nil)
                if cancel { precondition(failure is CancellationError) }
                else { precondition(failure?.localizedDescription.contains("catalog_has_no_public_records") == true) }
                precondition(!FileManager.default.fileExists(atPath: request.output))
                let staged = try FileManager.default.contentsOfDirectory(atPath: directory.appendingPathComponent("outputs").path)
                precondition(!staged.contains { $0.hasPrefix(".tc-image-staging-") })
                let restored = NativeJobStore(directory: directory)
                precondition(!restored.busy && restored.jobs[0].id == job.id && restored.jobs[0].state == job.state)
                print("App job \(model) \(cancel ? "cancel" : "reject") PASS")
            }
        }
    }
}
