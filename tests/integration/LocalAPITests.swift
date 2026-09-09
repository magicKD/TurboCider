import Foundation

@main struct LocalAPITests {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-api-test-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let socket = "/private/tmp/tc-api-\(UUID().uuidString.prefix(8)).sock"
        defer { try? FileManager.default.removeItem(atPath: socket + ".lock") }
        let executable = Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("turbocider")
        let store = NativeJobStore(directory: root.appendingPathComponent("studio"))
        let api = LocalAPIController(directory: root, executable: executable, socketPath: socket)
        defer { api.terminateNow() }
        func check(_ ok: Bool, _ text: String) throws { if !ok { throw NativeFailure(message: text) } }
        await api.start(store: store)
        try check(api.running && store.externalServiceActive, "App service did not start: \(api.error ?? "")")
        await api.refresh()
        try check(api.error == nil && api.jobCount == 0 && api.activeJob.isEmpty, "App service status failed")
        do {
            try await store.load(modelURL: root, modelID: "z-image-turbo")
            throw NativeFailure(message: "Embedded model loaded while API owned execution")
        } catch { try check(error.localizedDescription.contains("API"), "Wrong execution ownership error") }
        let other = LocalAPIController(directory: root.appendingPathComponent("other"), executable: executable, socketPath: socket)
        let otherStore = NativeJobStore(directory: root.appendingPathComponent("other-studio"))
        await other.start(store: otherStore)
        try check(!other.running && other.error != nil, "Second App attached to another process's socket")
        await api.refresh(); try check(api.error == nil, "Second start damaged the running service")
        api.stop()
        for _ in 0..<100 { if !api.changing && !api.running { break }; try await Task.sleep(for: .milliseconds(100)) }
        try check(!api.running && !store.externalServiceActive && !FileManager.default.fileExists(atPath: socket), "App stop left socket or execution reservation")
        await api.start(store: store)
        try check(api.running, "App service cannot restart: \(api.error ?? "")")
        api.stop()
        for _ in 0..<100 { if !api.changing && !api.running { break }; try await Task.sleep(for: .milliseconds(100)) }
        try check(!api.running && !store.externalServiceActive, "Restarted service did not release execution")
        let missing = LocalAPIController(directory: root.appendingPathComponent("missing"), executable: root.appendingPathComponent("missing-cli"), socketPath: socket)
        await missing.start(store: store)
        try check(!missing.running && !store.externalServiceActive && missing.error != nil, "Spawn failure left execution reserved")
        print("PASS: App API start/status/stop/restart, execution ownership, duplicate socket isolation, failed launch recovery")
    }
}
