import Foundation

@main struct ANELibraryTests {
    @MainActor static func main() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-ane-library-\(UUID())").resolvingSymlinksInPath()
        defer { try? fm.removeItem(at: root) }
        let previous = getenv("TURBOCIDER_MODEL_LIBRARY").map { String(cString: $0) }
        setenv("TURBOCIDER_MODEL_LIBRARY", root.appendingPathComponent("library").path, 1)
        defer { if let previous { setenv("TURBOCIDER_MODEL_LIBRARY", previous, 1) } else { unsetenv("TURBOCIDER_MODEL_LIBRARY") } }
        func check(_ value: Bool, _ message: String) throws { if !value { throw NativeFailure(message: message) } }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected invalid ANE registration to fail")
        }
        let model = root.appendingPathComponent("model")
        let checkpoint = model.appendingPathComponent("split_files/diffusion_models/z_image_turbo_bf16.safetensors")
        try fm.createDirectory(at: checkpoint.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 0, count: 16).write(to: checkpoint)
        let directory = root.appendingPathComponent("external-partitions")
        var artifacts: [String: [String: String]] = [:]
        for i in 0..<32 {
            let name = "block\(i).mlpackage"
            try fm.createDirectory(at: directory.appendingPathComponent(name), withIntermediateDirectories: true)
            artifacts[String(i)] = ["int8_pc": name]
        }
        let manifest = directory.appendingPathComponent("manifest.json")
        var raw: [String: Any] = ["schema_version": 2,
            "shape": ["K": 3840, "N": 3840, "input_mode": "enumerated", "buckets": [1056,1088,1536]],
            "source": ["checkpoint": checkpoint.path, "checkpoint_bytes": 16], "artifacts": artifacts]
        func write() throws { try JSONSerialization.data(withJSONObject: raw).write(to: manifest) }
        try write()
        let store = try LibraryStore()
        let item = try store.registerANE(modelID: "z-image-turbo", manifest: manifest)
        let alias = root.appendingPathComponent("alias.json")
        try fm.createSymbolicLink(at: alias, withDestinationURL: manifest)
        try check(try store.registerANE(modelID: "auto", manifest: alias).id == item.id, "Manifest alias was duplicated")
        let match = AccelerationDiscovery.find(modelPath: model.path, minimumRows: 1088, modelID: "z-image-turbo", requireCompiled: false)
        try check(match?.manifest == manifest.path && match?.rows == 1088, "Registered variable source was not discovered without draft paths")
        try check(AccelerationDiscovery.find(modelPath: model.path, minimumRows: 1568, modelID: "z-image-turbo", requireCompiled: false) == nil, "Capacity overflow accepted")
        try check(AccelerationDiscovery.find(modelPath: model.path, minimumRows: 1088, modelID: "z-image-turbo", loras: [StudioLoRA(path: "/missing.safetensors")], requireCompiled: false) == nil, "Base source accepted active LoRA")
        try rejects { _ = try store.registerANE(modelID: "flux2-klein-4b", manifest: manifest) }
        raw["shape"] = ["K":3840,"N":3840,"input_mode":"fixed","buckets":[1056,1088]]
        try write(); try rejects { _ = try store.registerANE(modelID: "z-image-turbo", manifest: manifest) }
        raw["shape"] = ["K":3840,"N":3840,"input_mode":"range","buckets":[1056,1088,1536]]
        try write(); let refreshed = try store.registerANE(modelID: "z-image-turbo", manifest: manifest)
        try check(refreshed.id == item.id && refreshed.inputMode == "range", "Re-registration did not refresh metadata")
        let config = root.appendingPathComponent("configuration.json")
        try JSONSerialization.data(withJSONObject: ["modelPaths": [:], "anePartitions": [["modelID": "z-image-turbo", "path": manifest.path]]]).write(to: config)
        _ = try await LibraryTool.run(["import", config.path, "--root", store.root.path])
        try check(try store.read().anePartitions?.count == 1, "Configuration import duplicated partition")
        try fm.removeItem(at: directory.appendingPathComponent("block0.mlpackage"))
        try check(AccelerationDiscovery.find(modelPath: model.path, minimumRows: 1088, modelID: "z-image-turbo", requireCompiled: false) == nil, "Stale registry bypassed artifact checks")
        try rejects { _ = try store.registerANE(modelID: "z-image-turbo", manifest: manifest) }
        try store.unregisterANE(id: item.id)
        try check(fm.fileExists(atPath: manifest.path) && fm.fileExists(atPath: checkpoint.path), "Unregister removed external files")
        try check(try store.read().anePartitions?.isEmpty == true, "Registration was not removed")
        print("PASS: ANE registration, alias deduplication, model/LoRA/capacity matching, dynamic source discovery, refresh, import, stale-artifact checks and non-destructive removal")
    }
}
