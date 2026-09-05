import Foundation

public enum TurboCiderDaemonError: LocalizedError {
    case packageRootNotFound
    case launchFailed(String)

    public var errorDescription: String? {
        switch self {
        case .packageRootNotFound:
            return "Could not locate the TurboCider package root. Set TURBOCIDER_ROOT."
        case .launchFailed(let detail):
            return "TurboCider daemon did not become ready: \(detail)"
        }
    }
}

public final class TurboCiderDaemon: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var logHandle: FileHandle?

    public init() {}

    public var ownedProcessID: Int32? {
        lock.withLock { process?.isRunning == true ? process?.processIdentifier : nil }
    }

    public func ensureRunning(
        client: TurboCiderClient = TurboCiderClient(),
        timeoutSeconds: Double = 15
    ) async throws {
        if (try? await client.health()) != nil { return }
        try start(client: client)
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if (try? await client.health()) != nil { return }
            if lock.withLock({ process?.isRunning == false }) {
                throw TurboCiderDaemonError.launchFailed("process exited during startup")
            }
            try await Task.sleep(nanoseconds: 200_000_000)
        }
        throw TurboCiderDaemonError.launchFailed("timed out after \(timeoutSeconds) seconds")
    }

    public func stop() {
        lock.lock()
        let current = process
        process = nil
        let handle = logHandle
        logHandle = nil
        lock.unlock()
        if current?.isRunning == true {
            current?.terminate()
            current?.waitUntilExit()
        }
        try? handle?.close()
    }

    deinit { stop() }

    public static func locatePackageRoot() -> URL? {
        let manager = FileManager.default
        var candidates: [URL] = []
        if let configured = ProcessInfo.processInfo.environment["TURBOCIDER_ROOT"] {
            candidates.append(URL(fileURLWithPath: configured, isDirectory: true))
        }
        candidates.append(URL(fileURLWithPath: manager.currentDirectoryPath, isDirectory: true))
        if let executable = Bundle.main.executableURL {
            let contents = executable.deletingLastPathComponent().deletingLastPathComponent()
            candidates.append(
                contents.appendingPathComponent("Resources/TurboCider", isDirectory: true)
            )
            var ancestor = executable.deletingLastPathComponent()
            for _ in 0..<8 {
                candidates.append(ancestor)
                ancestor.deleteLastPathComponent()
            }
        }
        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appendingPathComponent("TurboCider", isDirectory: true))
        }
        return candidates.first { candidate in
            manager.fileExists(atPath: candidate.appendingPathComponent("pyproject.toml").path)
                && manager.fileExists(
                    atPath: candidate.appendingPathComponent("src/turbocider/cli.py").path
                )
        }?.standardizedFileURL
    }

    private func start(client: TurboCiderClient) throws {
        lock.lock()
        if process?.isRunning == true {
            lock.unlock()
            return
        }
        lock.unlock()
        guard let root = Self.locatePackageRoot() else {
            throw TurboCiderDaemonError.packageRootNotFound
        }
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        )[0].appendingPathComponent("TurboCider", isDirectory: true)
        let logs = applicationSupport.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(
            at: logs, withIntermediateDirectories: true
        )
        let logURL = logs.appendingPathComponent("turbociderd.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()

        let launched = Process()
        let inheritedEnvironment = ProcessInfo.processInfo.environment
        let bundledCandidates = [
            root.appendingPathComponent("Python/bin/python3", isDirectory: false),
            root.appendingPathComponent("runtime/bin/python3", isDirectory: false),
            root.appendingPathComponent(".venv/bin/python3", isDirectory: false),
        ]
        let python = inheritedEnvironment["TURBOCIDER_PYTHON"]
            ?? bundledCandidates.first {
                FileManager.default.isExecutableFile(atPath: $0.path)
            }?.path
            ?? "/usr/bin/python3"
        launched.executableURL = URL(fileURLWithPath: python)
        let host = client.baseURL.host ?? "127.0.0.1"
        let port = client.baseURL.port ?? 11435
        guard ["127.0.0.1", "localhost", "::1"].contains(host) else {
            throw TurboCiderDaemonError.launchFailed(
                "automatic daemon startup requires a loopback API URL"
            )
        }
        launched.arguments = [
            "-m", "turbocider.cli", "serve",
            "--host", host, "--port", String(port),
        ]
        launched.currentDirectoryURL = root
        var environment = inheritedEnvironment
        if environment["TURBOCIDER_STATE_DIR"] == nil {
            environment["TURBOCIDER_STATE_DIR"] = applicationSupport
                .appendingPathComponent("state", isDirectory: true).path
        }
        if environment["TURBOCIDER_OUTPUT_DIR"] == nil {
            environment["TURBOCIDER_OUTPUT_DIR"] = applicationSupport
                .appendingPathComponent("outputs", isDirectory: true).path
        }
        if environment["TURBOCIDER_MODELS_DIR"] == nil {
            environment["TURBOCIDER_MODELS_DIR"] = applicationSupport
                .appendingPathComponent("models", isDirectory: true).path
        }
        if environment["TURBOCIDER_ENGINES_DIR"] == nil {
            environment["TURBOCIDER_ENGINES_DIR"] = root
                .appendingPathComponent("engines", isDirectory: true).path
        }
        let workspaceFile = root.appendingPathComponent("workspace.path")
        if environment["TURBOCIDER_WORKSPACE"] == nil,
           let value = try? String(contentsOf: workspaceFile, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
           !value.isEmpty {
            environment["TURBOCIDER_WORKSPACE"] = value
        }
        let source = root.appendingPathComponent("src").path
        if let existing = environment["PYTHONPATH"], !existing.isEmpty {
            environment["PYTHONPATH"] = source + ":" + existing
        } else {
            environment["PYTHONPATH"] = source
        }
        launched.environment = environment
        launched.standardOutput = handle
        launched.standardError = handle
        try launched.run()
        lock.withLock {
            process = launched
            logHandle = handle
        }
    }
}

private extension NSLock {
    func withLock<T>(_ operation: () -> T) -> T {
        lock()
        defer { unlock() }
        return operation()
    }
}
