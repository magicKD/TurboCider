import Foundation

@main struct LibraryStoreTests {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-library-test-\(UUID())").resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw LibraryFailure(message: message) }
        }
        func rejects(_ operation: () throws -> Void) throws {
            do { try operation() } catch { return }
            throw LibraryFailure(message: "Expected rejection")
        }
        let previousSettings = getenv("TURBOCIDER_LIBRARY_SETTINGS").map { String(cString: $0) }
        let previousRoot = getenv("TURBOCIDER_MODEL_LIBRARY").map { String(cString: $0) }
        setenv("TURBOCIDER_LIBRARY_SETTINGS", root.appendingPathComponent("settings.json").path, 1)
        unsetenv("TURBOCIDER_MODEL_LIBRARY")
        defer {
            if let previousSettings { setenv("TURBOCIDER_LIBRARY_SETTINGS", previousSettings, 1) } else { unsetenv("TURBOCIDER_LIBRARY_SETTINGS") }
            if let previousRoot { setenv("TURBOCIDER_MODEL_LIBRARY", previousRoot, 1) } else { unsetenv("TURBOCIDER_MODEL_LIBRARY") }
        }
        try LibrarySettings.configure(root.appendingPathComponent("configured"))
        try check(LibraryStore.defaultRoot.path == root.appendingPathComponent("configured").path, "Shared root configuration did not persist")
        let textRoot = root.appendingPathComponent("shared-text")
        let encoder = textRoot.appendingPathComponent("text_encoder"), tokenizer = textRoot.appendingPathComponent("tokenizer")
        try FileManager.default.createDirectory(at: encoder, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        let validConfig: [String: Any] = ["model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151936]
        try JSONSerialization.data(withJSONObject: validConfig).write(to: encoder.appendingPathComponent("config.json"))
        try Data("{\"tokenizer_class\":\"Qwen2Tokenizer\"}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer_config.json"))
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
        try Data(repeating: 0, count: 16).write(to: encoder.appendingPathComponent("model.safetensors"))
        try check(try SharedTextComponents.inspect(textRoot).count == 2, "Compatible text components rejected")
        var invalidConfig = validConfig; invalidConfig["hidden_size"] = 4096
        try JSONSerialization.data(withJSONObject: invalidConfig).write(to: encoder.appendingPathComponent("config.json"))
        try rejects { _ = try SharedTextComponents.inspect(textRoot) }
        try JSONSerialization.data(withJSONObject: validConfig).write(to: encoder.appendingPathComponent("config.json"))
        try Data("{\"weight_map\":{\"weight\":\"missing.safetensors\"}}".utf8).write(to: encoder.appendingPathComponent("model.safetensors.index.json"))
        try rejects { _ = try SharedTextComponents.inspect(textRoot) }
        let external = root.appendingPathComponent("external")
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        try Data("weights".utf8).write(to: external.appendingPathComponent("weights.bin"))
        let store = try LibraryStore(root: root.appendingPathComponent("library"))
        let registration = try store.register(modelID: "z-image-turbo", path: external)
        let duplicate = try store.register(modelID: "z-image-turbo", path: external)
        try check(registration.id == duplicate.id && !registration.managed, "External registration is not idempotent")
        let reopened = try LibraryStore(root: store.root)
        try check(try reopened.read().installations.count == 1, "Index did not persist")
        do {
            let lease = try store.acquireLease()
            try rejects { _ = try reopened.register(modelID: "flux2-klein-4b", path: external) }
            withExtendedLifetime(lease) {}
        }
        try store.unregister(id: registration.id)
        try check(FileManager.default.fileExists(atPath: external.appendingPathComponent("weights.bin").path), "Unregister deleted external weights")
        try check(try store.read().installations.isEmpty, "Unregister did not update index")
        for path in ["../escape", "/absolute", "a/../../b", "a//b", "a\\b", "a\nfile"] {
            try rejects { try LibraryStore.validateRelativePath(path) }
        }
        let temporary = root.appendingPathComponent("download")
        let bytes = Data("tiny verified artifact".utf8)
        try bytes.write(to: temporary)
        let hash = try LibraryStore.sha256(temporary)
        let lease = try store.acquireLease(); defer { withExtendedLifetime(lease) {} }
        try rejects { _ = try store.commitBlob(temporary: temporary, expectedSize: 1, expectedSHA256: hash) }
        try rejects { _ = try store.commitBlob(temporary: temporary, expectedSize: Int64(bytes.count), expectedSHA256: String(repeating: "0", count: 64)) }
        let blob = try store.commitBlob(temporary: temporary, expectedSize: Int64(bytes.count), expectedSHA256: hash)
        try bytes.write(to: temporary)
        let reused = try store.commitBlob(temporary: temporary, expectedSize: Int64(bytes.count), expectedSHA256: hash)
        try check(blob == reused && !FileManager.default.fileExists(atPath: temporary.path), "Cross-download blob reuse failed")
        let staging = try store.managedDirectory("staging").appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try LibraryStore.link(blob, at: "transformer/model.safetensors", in: staging)
        try LibraryStore.link(external, at: "text_encoder", in: staging)
        try rejects { try LibraryStore.link(blob, at: "text_encoder/overwrite.bin", in: staging) }
        try check(!FileManager.default.fileExists(atPath: external.appendingPathComponent("overwrite.bin").path), "Link collision wrote into an external component")
        let installed = try store.publish(staging: staging, modelID: "z-image-turbo", name: "Fixture", provider: "fixture", repository: "test/tiny", revision: "test-sha", components: [:])
        try check(installed.managed, "Published installation not marked managed")
        try check(try Data(contentsOf: URL(fileURLWithPath: installed.path).appendingPathComponent("transformer/model.safetensors")) == bytes, "Published blob link is broken")
        try check(try store.read().installations.count == 1, "Published installation is missing")
        try rejects { _ = try store.managedDirectory("../external") }
        try FileManager.default.createSymbolicLink(at: store.root.appendingPathComponent("unsafe"), withDestinationURL: external)
        try rejects { _ = try store.managedDirectory("unsafe") }
        print("PASS: persistence, external registration, multiprocess lock, path validation, SHA/size integrity, content deduplication, shared links, atomic publication")
    }
}
