import Foundation

/// Bounded local discovery. Never walks a home directory or downloads artifacts.
struct AccelerationDiscovery {
    struct Match {
        var manifest: String
        var source: String
        var rows: Int
        var mlpWidth: Int
        var aneMLPStart: Int
        var aneMLPEnd: Int
    }
    private static func currentHardware() -> (String, UInt64) {
        guard let data = NativeEngine.system().data(using: .utf8),
              let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return ("unknown", 0) }
        return (value["gpu"] as? String ?? "unknown",
                (value["physical_memory_bytes"] as? NSNumber)?.uint64Value ?? 0)
    }
    static func manifestBinds(manifest: String, loras: [StudioLoRA]) -> Bool {
        guard !manifest.isEmpty, !loras.isEmpty,
              let data = try? Data(contentsOf: URL(fileURLWithPath: manifest)),
              let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              value["schema_version"] as? Int == 2,
              let source = value["source"] as? [String: Any],
              let identities = source["loras"] as? [[String: Any]],
              identities.count == loras.count else { return false }
        let fm = FileManager.default
        for (identity, lora) in zip(identities, loras) {
            guard let recordedPath = identity["path"] as? String,
                  let recordedBytes = identity["bytes"] as? NSNumber,
                  let recordedSHA = identity["sha256"] as? String,
                  recordedSHA.count == 64,
                  recordedSHA.allSatisfy({ $0.isHexDigit }),
                  let recordedRole = identity["role"] as? String,
                  let recordedStrength = identity["strength"] as? NSNumber,
                  recordedRole == lora.role,
                  abs(recordedStrength.doubleValue - lora.strength) <= 1e-7 else {
                return false
            }
            let active = URL(fileURLWithPath: lora.path).resolvingSymlinksInPath().standardizedFileURL
            let recorded = URL(fileURLWithPath: recordedPath).resolvingSymlinksInPath().standardizedFileURL
            guard active == recorded,
                  let size = (try? fm.attributesOfItem(atPath: active.path))?[.size] as? NSNumber,
                  size.uint64Value == recordedBytes.uint64Value else { return false }
        }
        return true
    }
    static func automaticPolicyMatches(gpu: String, memory: UInt64,
                                       mlpWidth: Int, start: Int, end: Int,
                                       modelID: String = "flux2-klein-4b") -> Bool {
        // Automatic selection is a performance promise, not merely a
        // capability check. Keep candidates opt-in until their warm
        // end-to-end path beats the GPU baseline on validated hardware.
        guard modelID == "flux2-klein-4b" else { return false }
        if gpu == "Apple M4 Max" && memory == 64 * 1024 * 1024 * 1024 {
            return mlpWidth == 9216 && start == 0 && end == 6144
        }
        if gpu == "Apple M4 Pro" && memory == 48 * 1024 * 1024 * 1024 {
            return mlpWidth == 9216 && start == 0 && end == 9216
        }
        return false
    }
    static func find(modelPath: String, preferred: String = "", cache: URL? = nil,
                     enforceAutomaticPolicy: Bool = false,
                     modelID: String = "flux2-klein-4b") -> Match? {
        guard !modelPath.isEmpty else { return nil }
        let hardware = enforceAutomaticPolicy ? currentHardware() : nil
        let model = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let checkpointCandidates: [URL]
        if modelID == "z-image-turbo" {
            checkpointCandidates = [
                model.appendingPathComponent("split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
                model.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors.index.json"),
                model.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors")
            ]
        } else {
            checkpointCandidates = [model.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors")]
        }
        guard let checkpoint = checkpointCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return nil }
        guard let size = (try? FileManager.default.attributesOfItem(atPath: checkpoint.path))?[.size] as? NSNumber else { return nil }
        let fm = FileManager.default
        let appCache = cache ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative/cache/coreml")
        var candidates: [URL] = []
        if !preferred.isEmpty, URL(fileURLWithPath: preferred).resolvingSymlinksInPath().path.hasPrefix(appCache.resolvingSymlinksInPath().path + "/") { candidates.append(URL(fileURLWithPath: preferred)) }
        if let configured = ProcessInfo.processInfo.environment["TURBOCIDER_ANE_MANIFEST"] { candidates.append(URL(fileURLWithPath: configured)) }
        candidates += ((try? fm.contentsOfDirectory(at: appCache, includingPropertiesForKeys: nil)) ?? []).filter { $0.lastPathComponent.hasPrefix("manifest-") && $0.pathExtension == "json" }.sorted { $0.path < $1.path }
        for file in candidates {
            guard let data = try? Data(contentsOf: file),
                  let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  value["schema_version"] as? Int == 2,
                  let shape = value["shape"] as? [String: Any],
                  let hidden = (shape["K"] as? NSNumber)?.intValue,
                  (shape["N"] as? NSNumber)?.intValue == hidden,
                  let buckets = shape["buckets"] as? [Int], buckets.count == 1, (1...8192).contains(buckets[0]),
                  let source = value["source"] as? [String: Any], let path = source["checkpoint"] as? String,
                  URL(fileURLWithPath: path).resolvingSymlinksInPath() == checkpoint,
                  (source["checkpoint_bytes"] as? NSNumber)?.uint64Value == size.uint64Value,
                  let artifacts = value["artifacts"] as? [String: [String: String]] else { continue }
            let mlpWidth = (shape["mlp_width"] as? NSNumber)?.intValue ?? 9216
            let aneMLPStart = (shape["ane_mlp_start"] as? NSNumber)?.intValue ?? 0
            let aneMLPEnd = (shape["ane_mlp_end"] as? NSNumber)?.intValue ?? mlpWidth
            if let hardware {
                guard automaticPolicyMatches(gpu: hardware.0, memory: hardware.1,
                                             mlpWidth: mlpWidth,
                                             start: aneMLPStart, end: aneMLPEnd,
                                             modelID: modelID) else { continue }
            }
            let parent = file.deletingLastPathComponent().resolvingSymlinksInPath()
            let expectedBlocks = modelID == "z-image-turbo" ? 32 : 20
            guard hidden == (modelID == "z-image-turbo" ? 3840 : 3072) else { continue }
            let complete = (0..<expectedBlocks).allSatisfy { index in
                guard let relative = artifacts[String(index)]?["int8_pc"] else { return false }
                let artifact = parent.appendingPathComponent(relative).resolvingSymlinksInPath()
                var directory: ObjCBool = false
                return artifact.path.hasPrefix(parent.path + "/") && artifact.pathExtension == "mlmodelc" && fm.fileExists(atPath: artifact.path, isDirectory: &directory) && directory.boolValue
            }
            guard complete else { continue }
            let sourceFile = (value["source_manifest"] as? String).map { URL(fileURLWithPath: $0) } ?? parent.deletingLastPathComponent().appendingPathComponent("manifest.json")
            return Match(manifest: file.path,
                         source: fm.fileExists(atPath: sourceFile.path) ? sourceFile.path : "",
                         rows: buckets[0], mlpWidth: mlpWidth,
                         aneMLPStart: aneMLPStart, aneMLPEnd: aneMLPEnd)
        }
        return nil
    }
}
