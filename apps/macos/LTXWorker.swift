import Foundation
import Darwin

/// The CLI owns the disposable denoiser process and execs the clean VAE
/// finalizer. Never enable EXEC_FINALIZER in the desktop process itself.
enum LTXWorker {
    static func accepts(_ request: NativeRequest) -> Bool {
        request.model == "ltx-2.5-distilled" && request.residency == "component_staged"
    }

    static func generate(model: URL, request: NativeRequest, executable: URL? = nil,
                         onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        let worker = Task.detached(priority: .userInitiated) { () throws -> Data in
            let fm = FileManager.default
            let directory = fm.temporaryDirectory.appendingPathComponent("tc-ltx-app-\(UUID())")
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: directory) }
            let input = directory.appendingPathComponent("request.json")
            try JSONEncoder().encode(request).write(to: input, options: .atomic)
            let output = directory.appendingPathComponent("result.json")
            let events = directory.appendingPathComponent("events.jsonl")
            fm.createFile(atPath: output.path, contents: nil)
            fm.createFile(atPath: events.path, contents: nil)
            let stdout = try FileHandle(forWritingTo: output)
            let stderr = try FileHandle(forWritingTo: events)
            let reader = try FileHandle(forReadingFrom: events)
            defer { try? stdout.close(); try? stderr.close(); try? reader.close() }
            let child = Process()
            child.executableURL = executable ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("turbocider")
            child.arguments = ["generate", model.path, input.path]
            child.standardOutput = stdout; child.standardError = stderr
            var environment = ProcessInfo.processInfo.environment
            environment["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = environment["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] ?? TensorCache.sharedRoot.path
            child.environment = environment
            try Task.checkCancellation()
            try child.run()
            defer { if child.isRunning { kill(child.processIdentifier, SIGKILL) } }
            var pending = Data(), failure = "LTX 视频生成失败。"
            var cancelledAt: ContinuousClock.Instant?
            func drain() throws {
                while let data = try reader.read(upToCount: 256 * 1024), !data.isEmpty {
                    pending.append(data)
                    while let end = pending.firstIndex(of: 0x0a) {
                        let line = Data(pending[..<end]); pending.removeSubrange(...end)
                        if let event = try? JSONDecoder().decode(NativeEvent.self, from: line) { onEvent(event) }
                        else if !line.isEmpty { failure = String(decoding: line.suffix(4000), as: UTF8.self) }
                    }
                    if pending.count > 64 * 1024 { pending = Data(pending.suffix(64 * 1024)) }
                }
            }
            while child.isRunning {
                try drain()
                if Task.isCancelled, cancelledAt == nil { cancelledAt = .now; child.terminate() }
                if let cancelledAt, cancelledAt.duration(to: .now) > .seconds(5) {
                    kill(child.processIdentifier, SIGKILL)
                }
                // Preserve the cancellation grace period even after task cancellation.
                await Task.detached { try? await Task.sleep(for: .milliseconds(100)) }.value
            }
            child.waitUntilExit()
            try drain()
            if Task.isCancelled || child.terminationStatus == 2 { throw CancellationError() }
            if !pending.isEmpty { failure = String(decoding: pending.suffix(4000), as: UTF8.self) }
            guard child.terminationStatus == 0 else { throw NativeFailure(message: failure) }
            let result = try Data(contentsOf: output)
            guard (try? JSONSerialization.jsonObject(with: result)) is [String: Any],
                  fm.fileExists(atPath: request.output) else {
                throw NativeFailure(message: "LTX 未返回完整的视频结果，请检查模型和 App 安装。")
            }
            return result
        }
        return try await withTaskCancellationHandler { try await worker.value } onCancel: { worker.cancel() }
    }
}
