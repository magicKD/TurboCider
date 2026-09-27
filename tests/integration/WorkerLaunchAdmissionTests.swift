import Foundation
import Darwin

@main struct WorkerLaunchAdmissionTests {
    static func main() async throws {
        let executable = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-admission-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func waitReady(_ path: URL) async throws -> [Int32] {
            for _ in 0..<200 {
                if let value = try? String(contentsOf: path, encoding: .utf8), value.hasSuffix("\n") {
                    return value.split(whereSeparator: { $0.isWhitespace }).compactMap { Int32($0) }
                }
                try await Task.sleep(for: .milliseconds(25))
            }
            fatalError("Missing ready marker")
        }
        let runner = NativeProcessRunner()
        let admission = WorkerLaunchAdmission(journal: root.appendingPathComponent("launch.json"), jobID: UUID(), requestID: UUID(), requestDigest: String(repeating: "a", count: 64))
        let ready = root.appendingPathComponent("ready")
        let task = Task { try await runner.run(executable: executable, arguments: ["cooperative", ready.path], admission: admission) }
        let ids = try await waitReady(ready)
        let data = try Data(contentsOf: admission.journal)
        let record = try JSONDecoder().decode(WorkerLaunchAdmission.Record.self, from: data)
        precondition(record.jobID == admission.jobID && record.requestID == admission.requestID && record.requestDigest == admission.requestDigest)
        precondition(admission.observe() == .present)
        let wrongRequest = WorkerLaunchAdmission(journal: admission.journal, jobID: admission.jobID, requestID: UUID(), requestDigest: admission.requestDigest)
        precondition(wrongRequest.observe() == .unknown)
        precondition(record.process.pid == ids[0] && record.process.processGroup == ids[1] && record.process.observe() == .present)
        let wrongBirth = WorkerProcessIdentity(pid: ids[0], processGroup: ids[1], startSeconds: record.process.startSeconds + 1,
            startMicroseconds: record.process.startMicroseconds, bootSession: record.process.bootSession)
        precondition(wrongBirth.observe() == .unknown)
        task.cancel(); let result = try await task.value
        precondition(result.exitCode == 2 && !result.cleanupPending && result.failure == nil && record.process.observe() == .exited && admission.observe() == .exited)
        print("journal present before model admission; live/birth mismatch/reaped restart observations PASS")
        let replayMarker = root.appendingPathComponent("replay")
        let replay = try await runner.run(executable: executable, arguments: ["normal", replayMarker.path], admission: admission)
        precondition(replay.failure?.hasPrefix("worker_admission_failed") == true && !replay.cleanupPending)
        let replayData = try Data(contentsOf: admission.journal)
        precondition(!FileManager.default.fileExists(atPath: replayMarker.path) && replayData == data)
        let bad = WorkerLaunchAdmission(journal: root.appendingPathComponent("absent/launch.json"), jobID: UUID(), requestID: UUID(), requestDigest: admission.requestDigest)
        precondition(bad.observe() == .unknown)
        let denied = try await runner.run(executable: executable, arguments: ["normal", root.appendingPathComponent("denied").path], admission: bad)
        precondition(denied.failure?.hasPrefix("worker_admission_failed") == true && !denied.cleanupPending)
        precondition(!FileManager.default.fileExists(atPath: root.appendingPathComponent("denied").path))
        print("stale journal not overwritten; persistence failure never admits worker PASS")
        let parentReady = root.appendingPathComponent("parent")
        let parent = Process(); parent.executableURL = executable; parent.arguments = ["parent_before_admission", parentReady.path]
        try parent.run()
        let parentIDs = try await waitReady(parentReady)
        precondition(parentIDs.count == 3 && parentIDs[0] == parent.processIdentifier)
        kill(parent.processIdentifier, SIGKILL); parent.waitUntilExit()
        var gone = false
        for _ in 0..<200 {
            if kill(parentIDs[2], 0) == -1 && errno == ESRCH { gone = true; break }
            try await Task.sleep(for: .milliseconds(25))
        }
        precondition(gone && !FileManager.default.fileExists(atPath: parentReady.path + ".child"))
        print("parent death before admission closes pipe; orphan worker exits without work PASS")
    }
}
