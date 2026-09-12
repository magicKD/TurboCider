import Foundation

@main struct LibraryToolTests {
    static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-library-tool-test-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executable = Bundle.main.executableURL!.deletingLastPathComponent().appendingPathComponent("turbocider-library")
        let raw = try await LibraryTool.run(["list", "--root", root.path], executable: executable)
        let index = try LibraryTool.decode(LibraryIndex.self, from: raw)
        guard index.installations.isEmpty else { throw NativeFailure(message: "UI helper did not use requested root") }
        let folder = root.appendingPathComponent("external"); try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let registered = try await LibraryTool.run(["register", "fixture", folder.path, "--root", root.path], executable: executable)
        let item = try LibraryTool.decode(LibraryInstallation.self, from: registered)
        guard item.path == folder.resolvingSymlinksInPath().path else { throw NativeFailure(message: "UI helper registration path mismatch") }
        _ = try await LibraryTool.run(["remove", item.id, "--root", root.path], executable: executable)
        let adapter = folder.appendingPathComponent("style.safetensors")
        try Data(repeating: 0, count: 16).write(to: adapter)
        let config = root.appendingPathComponent("models.json")
        let payload: [String: Any] = ["schemaVersion": 1, "modelPaths": ["z-image-turbo": folder.path],
                                     "loras": [["modelID": "z-image-turbo", "path": adapter.path]]]
        try JSONSerialization.data(withJSONObject: payload).write(to: config)
        _ = try await LibraryTool.run(["import", config.path, "--root", root.path], executable: executable)
        let imported = try LibraryTool.decode(LibraryIndex.self, from: await LibraryTool.run(["list", "--root", root.path], executable: executable))
        guard imported.installations.count == 1, imported.loras?.first?.modelID == "z-image-turbo" else {
            throw NativeFailure(message: "Model and LoRA configuration import lost associations")
        }
        let stub = root.appendingPathComponent("slow-helper")
        try "#!/bin/sh\nexec /bin/sleep 30\n".write(to: stub, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: stub.path)
        let start = ContinuousClock.now
        let work = Task { try await LibraryTool.run(["download"], executable: stub) }
        try await Task.sleep(for: .milliseconds(150)); work.cancel()
        do { _ = try await work.value; throw NativeFailure(message: "Cancelled UI helper completed") } catch is CancellationError {}
        guard start.duration(to: .now) < .seconds(7) else { throw NativeFailure(message: "UI helper cancellation exceeded deadline") }
        print("PASS: App helper transport, typed registration, root isolation, cancellation cleanup")
    }
}
