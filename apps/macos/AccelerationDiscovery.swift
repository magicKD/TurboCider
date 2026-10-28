import Foundation
import Darwin

/// Bounded local discovery. Never walks a home directory or downloads artifacts.
struct AccelerationDiscovery {
    // Consume the native device policy so App and CLI share one optimization
    // whitelist. Old engines or unknown flags conservatively keep legacy behavior.
    static func optimizationEnabled(_ name: String, systemJSON: String = NativeEngine.system()) -> Bool {
        guard let data = systemJSON.data(using: .utf8),
              let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let profile = value["optimization_profile"] as? [String: Any]
        else { return false }
        return profile[name] as? Bool ?? false
    }
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
        var tiles: Int = 1
        var evaluatedRows: Int { rows * tiles }
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
    static func checkpoint(modelPath: String, modelID: String) -> URL? {
        guard !modelPath.isEmpty else { return nil }
        let model = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath()
        let paths: [String]
        switch modelID {
        case "z-image-turbo":
            paths = ["split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                     "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
                     "transformer/diffusion_pytorch_model.safetensors.index.json",
                     "transformer/diffusion_pytorch_model.safetensors"]
        case "qwen-image-2.1": paths = ["diffusion_models/qwen_image_2.1_bf16.safetensors"]
        default: paths = ["transformer/diffusion_pytorch_model.safetensors"]
        }
        return paths.map { model.appendingPathComponent($0) }
            .first { FileManager.default.fileExists(atPath: $0.path) }?.resolvingSymlinksInPath()
    }
    static func registeredCompiled(_ partitions: [LibraryANEPartition],
                                   modelID: String, modelPath: String) -> [LibraryANEPartition] {
        guard let active = checkpoint(modelPath: modelPath, modelID: modelID) else { return [] }
        return partitions.filter {
            $0.modelID == modelID && $0.kind == "compiled" &&
                sameCheckpoint(URL(fileURLWithPath: $0.checkpoint), active)
        }.sorted { $0.path < $1.path }
    }
    // This reads the selected artifact's format, not an optimization toggle.
    // Native loading still verifies all blocks, checkpoint SHA and calibration.
    static func qwen21ManifestUsesW8A8(_ path: String) throws -> Bool {
        let file = URL(fileURLWithPath: path)
        guard let size = fileIdentity(file)?.bytes, size <= 8 * 1024 * 1024,
              let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any],
              raw["schema_version"] as? Int == 2,
              let identity = raw["export_identity"] as? [String: Any],
              identity["owner"] as? String == "turbocider.qwen21.dit.coreml.v1",
              identity["tensor_layout"] as? String == "qwen21" else {
            throw NativeFailure(message: "无法识别 Qwen ANE manifest；请重新选择完整编译分区。")
        }
        if identity["variant"] as? String == "int8_pc", identity["activation_precision"] as? String == "int8" { return true }
        if identity["variant"] as? String == "fp16", identity["activation_precision"] as? String == "fp16" { return false }
        throw NativeFailure(message: "Qwen ANE manifest 的精度不受支持，请选择 W8A8 或 FP16 分区。")
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
        if gpu == "Apple M5 Pro" && memory == 24 * 1024 * 1024 * 1024 {
            return modelID == "flux2-klein-4b" &&
                mlpWidth == 9216 && start == 0 && end == 6144 &&
                (bucket == nil || bucket == 1088)
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
                     requireCompiled: Bool = true,
                     preferSmallestRows: Bool = false,
                     allowRowTiling: Bool = false) -> Match? {
        guard !modelPath.isEmpty else { return nil }
        // Adapter-bound partitions remain explicit until each LoRA geometry
        // has its own repeated warm end-to-end validation.
        if enforceAutomaticPolicy && !loras.isEmpty {
            return nil
        }
        let hardware = enforceAutomaticPolicy ? currentHardware() : nil
        guard let checkpoint = checkpoint(modelPath: modelPath, modelID: modelID) else { return nil }
        guard let size = fileIdentity(checkpoint)?.bytes else { return nil }
        let fm = FileManager.default
        let appCache = cache ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative/cache/coreml")
        var candidates: [URL] = []
        // Registered/offline-compiled partitions can live outside the App cache.
        // Automatic eligibility still requires the hardware, geometry, checkpoint,
        // LoRA and compilation checks below; a registration alone is insufficient.
        let allowExternal = !enforceAutomaticPolicy || optimizationEnabled("external_automatic_partitions")
        if !preferred.isEmpty, allowExternal || URL(fileURLWithPath: preferred).resolvingSymlinksInPath().path.hasPrefix(appCache.resolvingSymlinksInPath().path + "/") {
            candidates.append(URL(fileURLWithPath: preferred))
        }
        if allowExternal {
            candidates += knownManifests.map { URL(fileURLWithPath: $0) }
            for item in LibraryANEPartition.registered(modelID: modelID) {
                candidates.append(URL(fileURLWithPath: item.path))
                if !requireCompiled, let source = item.sourceManifest, !source.isEmpty {
                    candidates.append(URL(fileURLWithPath: source))
                }
            }
        }
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
                  let selectedRows = buckets.first(where: { $0 >= minimumRows }) ??
                    (allowRowTiling && modelID == "z-image-turbo" && loras.isEmpty &&
                     (value["export_identity"] as? [String: Any])?["image_only_token_rows"] == nil &&
                     ((value["export_identity"] as? [String: Any])?["mlp_output_kind"] as? String ?? "").isEmpty &&
                     (1...2048).contains(minimumRows) ? buckets.last : nil),
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
                                 aneMLPStart: aneMLPStart, aneMLPEnd: aneMLPEnd,
                                 tiles: max(1, (minimumRows + selectedRows - 1) / selectedRows)))
        }
        let preferredMatch = matches.first { $0.manifest == preferred }
        guard preferSmallestRows else {
            return preferredMatch ?? matches.min { $0.rows < $1.rows }
        }
        // A previous 1024 image must not force a 512 request to pad to 4128
        // rows when an equivalent 1120-row partition is available. Preserve
        // the selected MLP channel split; changing image size should not also
        // change the user's GPU/ANE division of work.
        let compatible = preferredMatch.map { selected in
            matches.filter {
                $0.mlpWidth == selected.mlpWidth &&
                $0.aneMLPStart == selected.aneMLPStart && $0.aneMLPEnd == selected.aneMLPEnd
            }
        } ?? matches
        return compatible.min {
            if $0.evaluatedRows != $1.evaluatedRows { return $0.evaluatedRows < $1.evaluatedRows }
            if $0.tiles != $1.tiles { return $0.tiles < $1.tiles }
            if ($0.manifest == preferred) != ($1.manifest == preferred) {
                return $0.manifest == preferred
            }
            return $0.manifest < $1.manifest
        }
    }
}
