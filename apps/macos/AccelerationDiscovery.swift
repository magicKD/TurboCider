import Foundation
import Darwin

/// Bounded local discovery. Never walks a home directory or downloads artifacts.
struct AccelerationDiscovery {
    private static func fileIdentity(_ url: URL) -> (bytes: UInt64, inode: UInt64, device: Int32)? {
        // Foundation attributesOfItem also queries extended attributes. Cache
        // discovery only needs stat fields; a file-provider getxattr can stall.
        var info = stat()
        guard fstatat(AT_FDCWD, url.path, &info, 0) == 0, info.st_size >= 0,
              info.st_mode & S_IFMT == S_IFREG else { return nil }
        return (UInt64(info.st_size), UInt64(info.st_ino), info.st_dev)
    }
    struct Match: Sendable {
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
    private static func sameCheckpoint(_ recorded: URL, _ active: URL) -> Bool {
        if recorded.resolvingSymlinksInPath() == active.resolvingSymlinksInPath() { return true }
        // Existing installations can bind the same checkpoint through hard links.
        guard let a = fileIdentity(recorded), let b = fileIdentity(active) else { return false }
        return a.inode == b.inode && a.device == b.device
    }
    static func manifestBinds(manifest: String, loras: [StudioLoRA]) -> Bool {
        guard !manifest.isEmpty, !loras.isEmpty,
              let data = try? Data(contentsOf: URL(fileURLWithPath: manifest)),
              let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              value["schema_version"] as? Int == 2,
              let source = value["source"] as? [String: Any],
              let identities = source["loras"] as? [[String: Any]],
              identities.count == loras.count else { return false }
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
                  let size = fileIdentity(active)?.bytes,
                  size == recordedBytes.uint64Value else { return false }
        }
        return true
    }
    static func automaticPolicyMatches(gpu: String, memory: UInt64,
                                       mlpWidth: Int, start: Int, end: Int,
                                       modelID: String = "flux2-klein-4b",
                                       bucket: Int? = nil) -> Bool {
        // Automatic selection is a performance promise, not merely a
        // capability check. Keep candidates opt-in until their warm
        // end-to-end path beats the GPU baseline on validated hardware.
        if gpu == "Apple M4 Max" && memory == 64 * 1024 * 1024 * 1024 {
            if modelID == "flux2-klein-4b" {
                return mlpWidth == 9216 && start == 0 && end == 6144 &&
                    (bucket == nil || bucket == 1088)
            }
            if modelID == "z-image-turbo" {
                return mlpWidth == 10240 && start == 0 && end == 4096 &&
                    (bucket == nil || bucket == 4128)
            }
            return false
        }
        if gpu == "Apple M4 Pro" && memory == 48 * 1024 * 1024 * 1024 {
            return modelID == "flux2-klein-4b" &&
                mlpWidth == 9216 && start == 0 && end == 9216 &&
                (bucket == nil || bucket == 1088 || bucket == 4160)
        }
        return false
    }
    static func automaticBucket(modelID: String, operation: String,
                                width: Int, height: Int, steps: Int,
                                residency: String, hasInputs: Bool) -> Int? {
        guard operation == "image.generate", width == height,
              residency == "resident", !hasInputs else { return nil }
        if modelID == "flux2-klein-4b" && steps == 4 {
            if width == 512 { return 1088 }
            if width == 1024 { return 4160 }
        }
        if modelID == "z-image-turbo" && width == 1024 && steps == 9 {
            return 4128
        }
        return nil
    }
    static func find(modelPath: String, preferred: String = "", cache: URL? = nil,
                     minimumRows: Int = 0,
                     requiredRows: Int? = nil,
                     enforceAutomaticPolicy: Bool = false,
                     modelID: String = "flux2-klein-4b",
                     loras: [StudioLoRA] = [], knownManifests: [String] = [],
                     requireCompiled: Bool = true) -> Match? {
        guard !modelPath.isEmpty else { return nil }
        // Adapter-bound partitions remain explicit until each LoRA geometry
        // has its own repeated warm end-to-end validation.
        if enforceAutomaticPolicy && !loras.isEmpty {
            return nil
        }
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
        guard let located = checkpointCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else { return nil }
        let checkpoint = located.resolvingSymlinksInPath()
        guard let size = fileIdentity(checkpoint)?.bytes else { return nil }
        let fm = FileManager.default
        let appCache = cache ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative/cache/coreml")
        var candidates: [URL] = []
        if !preferred.isEmpty, !enforceAutomaticPolicy || URL(fileURLWithPath: preferred).resolvingSymlinksInPath().path.hasPrefix(appCache.resolvingSymlinksInPath().path + "/") { candidates.append(URL(fileURLWithPath: preferred)) }
        if !enforceAutomaticPolicy { candidates += knownManifests.map { URL(fileURLWithPath: $0) } }
        if let configured = ProcessInfo.processInfo.environment["TURBOCIDER_ANE_MANIFEST"] { candidates.append(URL(fileURLWithPath: configured)) }
        candidates += ((try? fm.contentsOfDirectory(at: appCache, includingPropertiesForKeys: nil)) ?? []).filter { $0.lastPathComponent.hasPrefix("manifest-") && $0.pathExtension == "json" }.sorted { $0.path < $1.path }
        var matches: [Match] = []
        for file in candidates {
            guard let data = try? Data(contentsOf: file),
                  let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  value["schema_version"] as? Int == 2,
                  let shape = value["shape"] as? [String: Any],
                  let hidden = (shape["K"] as? NSNumber)?.intValue,
                  (shape["N"] as? NSNumber)?.intValue == hidden,
                  let buckets = shape["buckets"] as? [Int], !buckets.isEmpty, buckets.count <= 128,
                  buckets.allSatisfy({ (1...8192).contains($0) }),
                  buckets == Array(Set(buckets)).sorted(),
                  (buckets.count == 1 || ["enumerated", "range"].contains(shape["input_mode"] as? String ?? "")),
                  let selectedRows = buckets.first(where: { $0 >= minimumRows }),
                  let source = value["source"] as? [String: Any], let path = source["checkpoint"] as? String,
                  sameCheckpoint(URL(fileURLWithPath: path), checkpoint),
                  (source["checkpoint_bytes"] as? NSNumber)?.uint64Value == size,
                  let artifacts = value["artifacts"] as? [String: [String: String]] else { continue }
            if let requiredRows, selectedRows != requiredRows { continue }
            let declaredLoRAs = source["loras"] as? [[String: Any]]
            if loras.isEmpty {
                if declaredLoRAs?.isEmpty == false { continue }
            } else if !manifestBinds(manifest: file.path, loras: loras) {
                continue
            }
            let mlpWidth = (shape["mlp_width"] as? NSNumber)?.intValue ?? 9216
            let aneMLPStart = (shape["ane_mlp_start"] as? NSNumber)?.intValue ?? 0
            let aneMLPEnd = (shape["ane_mlp_end"] as? NSNumber)?.intValue ?? mlpWidth
            if let hardware {
                guard automaticPolicyMatches(gpu: hardware.0, memory: hardware.1,
                                             mlpWidth: mlpWidth,
                                             start: aneMLPStart, end: aneMLPEnd,
                                             modelID: modelID,
                                             bucket: selectedRows) else { continue }
            }
            let parent = file.deletingLastPathComponent().resolvingSymlinksInPath()
            let expectedBlocks = modelID == "z-image-turbo" ? 32 : 20
            guard hidden == (modelID == "z-image-turbo" ? 3840 : 3072) else { continue }
            let complete = (0..<expectedBlocks).allSatisfy { index in
                guard let relative = artifacts[String(index)]?["int8_pc"] else { return false }
                let artifact = parent.appendingPathComponent(relative).resolvingSymlinksInPath()
                var directory: ObjCBool = false
                guard artifact.path.hasPrefix(parent.path + "/"), fm.fileExists(atPath: artifact.path, isDirectory: &directory) else { return false }
                if !requireCompiled { return ["mlpackage", "mlmodel"].contains(artifact.pathExtension) }
                guard artifact.pathExtension == "mlmodelc", directory.boolValue else { return false }
                let identity = artifact.deletingLastPathComponent().appendingPathComponent("identity.json")
                if fm.fileExists(atPath: identity.path) {
                    guard let bytes = try? Data(contentsOf: identity),
                          let value = (try? JSONSerialization.jsonObject(with: bytes)) as? [String: Any],
                          let build = value["os_build"] as? String, !build.isEmpty,
                          ProcessInfo.processInfo.operatingSystemVersionString.contains(build),
                          value["gpu"] as? String == currentHardware().0,
                          value["architecture"] as? String == "arm64" else { return false }
                }
                return true
            }
            guard complete else { continue }
            let sourceFile = (value["source_manifest"] as? String).map { URL(fileURLWithPath: $0) } ?? parent.deletingLastPathComponent().appendingPathComponent("manifest.json")
            matches.append(Match(manifest: file.path,
                                 source: fm.fileExists(atPath: sourceFile.path) ? sourceFile.path : "",
                                 rows: selectedRows, mlpWidth: mlpWidth,
                                 aneMLPStart: aneMLPStart, aneMLPEnd: aneMLPEnd))
        }
        return matches.min { $0.rows < $1.rows }
    }
}
