import Foundation

/// Bounded local discovery. Never walks a home directory or downloads artifacts.
struct AccelerationDiscovery {
    struct Match { var manifest: String; var source: String; var rows: Int }
    static func find(modelPath: String, preferred: String = "", cache: URL? = nil, minimumRows: Int = 0) -> Match? {
        guard !modelPath.isEmpty else { return nil }
        let model = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let checkpoint = model.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors")
        guard let size = (try? FileManager.default.attributesOfItem(atPath: checkpoint.path))?[.size] as? NSNumber else { return nil }
        let fm = FileManager.default
        let appCache = cache ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative/cache/coreml")
        var candidates: [URL] = []
        if !preferred.isEmpty, URL(fileURLWithPath: preferred).resolvingSymlinksInPath().path.hasPrefix(appCache.resolvingSymlinksInPath().path + "/") { candidates.append(URL(fileURLWithPath: preferred)) }
        if let configured = ProcessInfo.processInfo.environment["TURBOCIDER_ANE_MANIFEST"] { candidates.append(URL(fileURLWithPath: configured)) }
        candidates += ((try? fm.contentsOfDirectory(at: appCache, includingPropertiesForKeys: nil)) ?? []).filter { $0.lastPathComponent.hasPrefix("manifest-") && $0.pathExtension == "json" }.sorted { $0.path < $1.path }
        var matches: [Match] = []
        for file in candidates {
            guard let data = try? Data(contentsOf: file),
                  let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  value["schema_version"] as? Int == 2,
                  let shape = value["shape"] as? [String: Any], shape["K"] as? Int == 3072, shape["N"] as? Int == 3072,
                  let buckets = shape["buckets"] as? [Int], buckets.count == 1, (1...8192).contains(buckets[0]), buckets[0] >= minimumRows,
                  let source = value["source"] as? [String: Any], let path = source["checkpoint"] as? String,
                  URL(fileURLWithPath: path).resolvingSymlinksInPath() == checkpoint,
                  (source["checkpoint_bytes"] as? NSNumber)?.uint64Value == size.uint64Value,
                  let artifacts = value["artifacts"] as? [String: [String: String]] else { continue }
            let parent = file.deletingLastPathComponent().resolvingSymlinksInPath()
            let complete = (0..<20).allSatisfy { index in
                guard let relative = artifacts[String(index)]?["int8_pc"] else { return false }
                let artifact = parent.appendingPathComponent(relative).resolvingSymlinksInPath()
                var directory: ObjCBool = false
                return artifact.path.hasPrefix(parent.path + "/") && artifact.pathExtension == "mlmodelc" && fm.fileExists(atPath: artifact.path, isDirectory: &directory) && directory.boolValue
            }
            guard complete else { continue }
            let sourceFile = (value["source_manifest"] as? String).map { URL(fileURLWithPath: $0) } ?? parent.deletingLastPathComponent().appendingPathComponent("manifest.json")
            matches.append(Match(manifest: file.path, source: fm.fileExists(atPath: sourceFile.path) ? sourceFile.path : "", rows: buckets[0]))
        }
        return matches.min { $0.rows < $1.rows }
    }
}
