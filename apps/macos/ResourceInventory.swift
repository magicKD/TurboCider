import Foundation
import Darwin

/// File-provider reads can block inside open(). Keep optional disk inspection out
/// of the inference process, and kill the helper at a bounded deadline.
enum ResourceInventory {
    static func run(_ payload: Data, timeout: Double = 5, executableURL: URL? = nil) async throws -> Data {
        try await Task.detached(priority: .utility) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("tc-inventory-\(UUID())")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let request = directory.appendingPathComponent("request.json")
            let output = directory.appendingPathComponent("output.json")
            let error = directory.appendingPathComponent("error.txt")
            try payload.write(to: request)
            _ = FileManager.default.createFile(atPath: output.path, contents: nil)
            _ = FileManager.default.createFile(atPath: error.path, contents: nil)
            let stdout = try FileHandle(forWritingTo: output), stderr = try FileHandle(forWritingTo: error)
            defer { try? stdout.close(); try? stderr.close() }
            guard let executable = executableURL ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("turbocider") else {
                throw NativeFailure(message: "找不到磁盘统计工具。")
            }
            let process = Process()
            process.executableURL = executable
            process.arguments = ["coreml", request.path]
            process.standardOutput = stdout; process.standardError = stderr
            try process.run()
            defer { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
            let start = ContinuousClock.now
            while process.isRunning {
                if start.duration(to: .now) > .seconds(timeout) {
                    kill(process.processIdentifier, SIGKILL)
                    throw NativeFailure(message: "磁盘统计超时，可能有不可访问的外部文件。生成与模型控制不受影响，可稍后刷新。")
                }
                try await Task.sleep(for: .milliseconds(50))
            }
            guard process.terminationStatus == 0 else {
                let message = (try? String(contentsOf: error, encoding: .utf8)) ?? "磁盘统计失败。"
                throw NativeFailure(message: message.isEmpty ? "磁盘统计未完成，可稍后重试。" : String(message.prefix(2000)))
            }
            return try Data(contentsOf: output)
        }.value
    }
}
