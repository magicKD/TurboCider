import Foundation

/// Real public CLI rejection/cancellation through the supervisor. Production
/// catalog remains empty; no successful model generation is claimed here.
@main struct NativeProcessWorkerTests {
    static func main() async throws {
        let executable = URL(fileURLWithPath: CommandLine.arguments[1])
        let manifest = URL(fileURLWithPath: CommandLine.arguments[2])
        let root = URL(fileURLWithPath: CommandLine.arguments[3])
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as! [String: Any]
        let fingerprint = metadata["runtime_build_id"] as! String
        let runner = NativeProcessRunner()
        for (model, installation) in [("z-image-turbo", "/Users/chencanhui/models/TurboCider/Z-Image-Turbo"),
                                      ("flux2-klein-4b", "/Users/chencanhui/models/TurboCider/FLUX.2-klein-4B")] {
            for cancel in [false, true] {
                let folder = root.appendingPathComponent(model + (cancel ? "-cancel" : "-reject"))
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
                var legacy = NativeRequest(prompt: "A red fox in snow", output: folder.appendingPathComponent("output.png").path)
                legacy.model = model; legacy.width = 512; legacy.height = 512; legacy.steps = model == "z-image-turbo" ? 9 : 4
                let request = NativeRequestV2(legacy: legacy, targetBytes: 10 << 30)
                let input = try WorkerRequestEnvelope.encode(jobID: UUID(), requestID: UUID(), installation: URL(fileURLWithPath: installation),
                    nativeRequest: JSONEncoder().encode(request))
                let inputPath = folder.appendingPathComponent("input.json")
                try input.write(to: inputPath, options: .withoutOverwriting)
                let plan: [String: Any] = ["runtime_fingerprint": fingerprint, "executable": executable.path,
                    "mode": "worker-generate", "expected": cancel ? "cooperative source cancellation, exit 2" : "catalog_has_no_public_records, exit 1",
                    "grace_seconds": 5, "reaping_seconds": 5, "cancel_delay_seconds": cancel ? 1 : 0,
                    "scope": "Real source check through native process supervisor; no successful model generation or timing qualification"]
                try JSONSerialization.data(withJSONObject: plan, options: .prettyPrinted).write(to: folder.appendingPathComponent("plan.json"))
                let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
                let admission = WorkerLaunchAdmission(journal: folder.appendingPathComponent("launch.json"),
                    jobID: UUID(uuidString: wire["job_id"] as! String)!, requestID: UUID(uuidString: wire["request_id"] as! String)!,
                    requestDigest: wire["request_digest"] as! String)
                let task = Task { try await runner.run(executable: executable, arguments: ["worker-generate", inputPath.path], admission: admission) }
                if cancel { try await Task.sleep(for: .seconds(1)); task.cancel() }
                let result = try await task.value
                try result.stdout.write(to: folder.appendingPathComponent("stdout.json"))
                try result.stderr.write(to: folder.appendingPathComponent("stderr.log"))
                let observation: [String: Any] = ["pid": result.pid, "exit_code": result.exitCode.map { $0 as Any } ?? NSNull(),
                    "cancellation_requested": result.cancellationRequested, "forced_stop": result.forcedStop,
                    "cleanup_pending": result.cleanupPending, "failure": result.failure.map { $0 as Any } ?? NSNull()]
                try JSONSerialization.data(withJSONObject: observation, options: .prettyPrinted).write(to: folder.appendingPathComponent("observation.json"))
                precondition(!result.cleanupPending && result.failure == nil && !result.forcedStop)
                precondition(result.cancellationRequested == cancel && result.exitCode == (cancel ? 2 : 1))
                let record = try JSONDecoder().decode(WorkerLaunchAdmission.Record.self, from: Data(contentsOf: admission.journal))
                precondition(record.process.pid == result.pid && record.process.observe() == .exited)
                let terminal = try WorkerTerminalEnvelope.validate(result.stdout, input: input, request: request,
                    runtimeFingerprint: fingerprint, operation: .generate, exitCode: result.exitCode!)
                precondition(terminal.status == (cancel ? "cancelled" : "error"))
                if !cancel { precondition(terminal.errorMessage == "catalog_has_no_public_records") }
                precondition(terminal.artifact == nil && !FileManager.default.fileExists(atPath: request.outputs[0].path))
                print("\(model) \(cancel ? "cancel" : "reject") supervisor + terminal PASS")
            }
        }
    }
}
