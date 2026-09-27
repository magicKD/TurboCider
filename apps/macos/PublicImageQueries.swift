import Foundation

/// Discovery never grants execution authority. A new generation resolves and
/// verifies its sources again in its own worker process.
enum PublicImageQueries {
    static func options(_ request: NativeRequestV2, modelURL: URL?) async throws -> NativeStreamingOptions {
        let candidates = try await Task.detached(priority: .utility) {
            try NativeEngine.workerStreamingOptions(request)
        }.value
        try Task.checkCancellation()
        guard let modelURL, candidates.targets.contains(where: { $0.status == "candidate" }) else { return candidates }
        return try await candidates.resolvingCandidates(for: request) {
            try await PublicImageQueryRunner.shared.resolve($0, model: modelURL)
        }
    }
}

/// Queries verify CPU/file sources only and never allocate model GPU weights.
/// Serialize these disk-heavy probes separately from the GPU generation runner.
actor PublicImageQueryRunner {
    static let shared = PublicImageQueryRunner()
    private let runner = NativeProcessRunner()
    private var active = false
    private var pendingDirectory: URL?

    func resolve(_ request: NativeRequestV2, model: URL, executable: URL? = nil) async throws -> NativeStreamingResolution {
        while active {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(25))
        }
        try Task.checkCancellation()
        // Claim before the next suspension point; queued queries cannot overlap.
        active = true
        defer { active = false }
        guard await runner.pollCleanup() else {
            throw NativeFailure(message: "worker_query_cleanup_pending", code: "worker_query_cleanup_pending")
        }
        if let pendingDirectory {
            try FileManager.default.removeItem(at: pendingDirectory)
            self.pendingDirectory = nil
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tc-image-query-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        var retain = false
        defer { if !retain { try? FileManager.default.removeItem(at: directory) } }
        let id = UUID()
        let prepared = try PublicImageWorker.prepare(jobID: id, model: model, request: request, store: directory)
        let executable = try executable ?? PublicImageWorker.executable()
        let admission = prepared.reference.admission(jobID: id, store: directory)
        let work = Task { try await runner.run(executable: executable,
            arguments: ["worker-query", prepared.inputURL.path], admission: admission) }
        let process = try await withTaskCancellationHandler {
            try await work.value
        } onCancel: { work.cancel() }
        if process.cleanupPending {
            retain = true; pendingDirectory = directory
            throw NativeFailure(message: "worker_query_cleanup_pending", code: "worker_query_cleanup_pending")
        }
        if process.cancellationRequested || Task.isCancelled { throw CancellationError() }
        if let failure = process.failure { throw NativeFailure(message: failure) }
        guard let exitCode = process.exitCode else { throw NativeFailure(message: "worker_query_terminated") }
        let terminal = try WorkerTerminalEnvelope.validate(process.stdout, input: prepared.input, request: request,
            runtimeFingerprint: prepared.reference.runtimeFingerprint, operation: .query, exitCode: exitCode)
        if terminal.status == "cancelled" { throw CancellationError() }
        guard terminal.status == "resolved", let resolution = terminal.resolution else {
            throw NativeFailure(message: terminal.errorMessage ?? "worker_query_invalid", code: terminal.errorCode)
        }
        return resolution
    }
}
