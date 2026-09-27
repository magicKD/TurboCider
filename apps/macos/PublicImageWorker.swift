import Foundation

/// Persist this reference with the pending job before invoking the runner.
/// Authority remains in the worker; the reference only binds recovery evidence.
enum PublicImageWorker {
    struct Reference: Codable, Sendable {
        let requestID: UUID
        let requestDigest: String
        let runtimeFingerprint: String
        let stagedOutput: String
        var started = false
        var exitConfirmed = false
        func directory(jobID: UUID, store: URL) -> URL {
            store.appendingPathComponent("worker-jobs", isDirectory: true)
                .appendingPathComponent(jobID.uuidString.lowercased(), isDirectory: true)
                .appendingPathComponent(requestID.uuidString.lowercased(), isDirectory: true)
        }
        func admission(jobID: UUID, store: URL) -> WorkerLaunchAdmission {
            WorkerLaunchAdmission(journal: directory(jobID: jobID, store: store).appendingPathComponent("launch.json"),
                jobID: jobID, requestID: requestID, requestDigest: requestDigest)
        }
    }
    struct Prepared: Sendable {
        let reference: Reference
        let input: Data
        let inputURL: URL
    }
    enum Failure: Error, LocalizedError {
        case cleanupPending
        var errorDescription: String? { "工作进程的退出尚未确认，清理完成前不能开始新的任务。" }
    }
    static func prepare(jobID: UUID, model: URL, request: NativeRequestV2, store: URL) throws -> Prepared {
        guard ["z-image-turbo", "flux2-klein-4b"].contains(request.model), request.operation == "image.generate", request.outputs.count == 1 else {
            throw NativeFailure(message: "worker_request_invalid: 不支持的图片请求。")
        }
        let requestID = UUID()
        let input = try WorkerRequestEnvelope.encode(jobID: jobID, requestID: requestID, installation: model,
                                                    nativeRequest: JSONEncoder().encode(request))
        let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
        let reference = Reference(requestID: requestID, requestDigest: wire["request_digest"] as! String,
            runtimeFingerprint: NativeEngine.runtimeBuildIdentity(), stagedOutput: request.outputs[0].path)
        guard !reference.runtimeFingerprint.isEmpty else { throw NativeFailure(message: "worker_runtime_identity_missing") }
        let directory = reference.directory(jobID: jobID, store: store)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let inputURL = directory.appendingPathComponent("input.json")
        try input.write(to: inputURL, options: .withoutOverwriting)
        return Prepared(reference: reference, input: input, inputURL: inputURL)
    }
    static func executable() throws -> URL {
        guard let app = Bundle.main.executableURL else { throw NativeFailure(message: "worker_executable_missing") }
        return app.deletingLastPathComponent().appendingPathComponent("turbocider")
    }
    static func saveDiagnostics(_ process: NativeProcessRunner.Result, reference: Reference, jobID: UUID, store: URL) throws {
        let directory = reference.directory(jobID: jobID, store: store)
        try process.stdout.write(to: directory.appendingPathComponent("stdout.json"), options: .atomic)
        try process.stderr.write(to: directory.appendingPathComponent("stderr.log"), options: .atomic)
    }
}
