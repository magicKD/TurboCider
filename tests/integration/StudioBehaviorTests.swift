import AppKit
import Combine
import Foundation
import CoreGraphics
import ImageIO

@MainActor private final class DeferredStudioImageProvider {
    private var reply: ((Data?, Error?) -> Void)?
    func makeProvider() -> NSItemProvider {
        let provider = NSItemProvider()
        provider.registerDataRepresentation(forTypeIdentifier: "public.png", visibility: .all) { reply in
            Task { @MainActor in self.reply = reply }
            return nil
        }
        return provider
    }
    func waitUntilRequested() async throws {
        let start = ContinuousClock.now
        while reply == nil {
            guard start.duration(to: .now) < .seconds(3) else {
                throw NativeFailure(message: "Deferred image provider was not requested")
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
    func finish(_ data: Data) { reply?(data, nil); reply = nil }
}

@main
struct StudioBehaviorTests {
    @MainActor static func main() async throws {
        if ProcessInfo.processInfo.environment["TURBOCIDER_TEST_ACCELERATION_ONLY"] == "1" {
            try await verifyAccelerationRouting()
            return
        }
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try condition() else { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected validation failure")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-studio-test-\(UUID())")
        let previousLibrary = getenv("TURBOCIDER_MODEL_LIBRARY").map { String(cString: $0) }
        setenv("TURBOCIDER_MODEL_LIBRARY", root.appendingPathComponent("model-library").path, 1)
        defer {
            if let previousLibrary { setenv("TURBOCIDER_MODEL_LIBRARY", previousLibrary, 1) }
            else { unsetenv("TURBOCIDER_MODEL_LIBRARY") }
        }
        defer { try? FileManager.default.removeItem(at: root) }
        let poisoned = NativeJobStore(directory: root.appendingPathComponent("poisoned-store"))
        try check(!poisoned.recordProcessQuarantine(NativeFailure(message: "ordinary failure")), "Ordinary errors must remain retryable")
        try check(poisoned.recordProcessQuarantine(NativeFailure(message: "streaming_process_quarantined: primary: cancelled; GPU drain incomplete")), "Quarantine error was ignored")
        try check(poisoned.requiresProcessRestart && !poisoned.canUnload, "Quarantined process remained usable")
        try check(poisoned.sessionState.contains("重启"), "Quarantine UI must require restart")
        do {
            try await poisoned.load(modelURL: root.appendingPathComponent("missing"))
            throw NativeFailure(message: "Quarantined store reopened an engine")
        } catch {
            try check(error.localizedDescription.contains("streaming_process_quarantined:"), "Quarantine lost on retry")
        }
        let monitor = ResourceMonitor()
        monitor.sample()
        try check(monitor.residentBytes.map { $0 > 0 } == true, "Resident memory unavailable")
        monitor.sample()
        try check(monitor.cpuPercent.map { (0...100).contains($0) } ?? true, "Invalid CPU counter")
        try check(monitor.gpuPercent.map { (0...100).contains($0) } ?? true, "Invalid GPU counter")
        let fifo = root.appendingPathComponent("blocked-manifest.json")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        guard mkfifo(fifo.path, 0o600) == 0 else { throw NativeFailure(message: "FIFO fixture failed") }
        let inventoryStore = NativeJobStore(directory: root.appendingPathComponent("inventory-store"))
        let inventoryPayload = try JSONSerialization.data(withJSONObject: ["action": "inventory", "cache": root.appendingPathComponent("empty-cache").path, "source_manifest": fifo.path])
        let inspection = Task { try await inventoryStore.coreMLResources(inventoryPayload) }
        while !inventoryStore.inspectingResources { await Task.yield() }
        try check(!inventoryStore.busy, "Disk inspection blocked generation controls")
        _ = try await inspection.value
        try check(!inventoryStore.busy && !inventoryStore.inspectingResources, "Inspection left controls locked")
        // Simulate a helper stuck in an external file provider. NSData rejects
        // FIFOs promptly on this OS, so a FIFO alone cannot exercise the deadline.
        let helper = root.appendingPathComponent("blocked-helper")
        try "#!/bin/sh\nexec /bin/sleep 30\n".write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
        let inventoryStart = ContinuousClock.now
        do {
            _ = try await ResourceInventory.run(inventoryPayload, timeout: 0.3, executableURL: helper)
            throw NativeFailure(message: "Blocked helper did not time out")
        } catch { try check(error.localizedDescription.contains("超时"), "Unexpected inspection failure: \(error)") }
        try check(inventoryStart.duration(to: .now) < .seconds(3), "Disk inspection exceeded deadline")
        let studio = StudioState(directory: root)
        let originalDraft = studio.draft
        let queryKey = studio.streamingQueryKey
        studio.draft.prompt += " changed"
        try check(studio.streamingQueryKey != queryKey, "Prompt must invalidate token-dependent options")
        studio.draft = originalDraft
        studio.draft.modelPaths[studio.draft.modelID] = "/changed/installation"
        try check(studio.streamingQueryKey != queryKey, "Installation must invalidate options")
        studio.draft = originalDraft
        studio.draft.dynamicText.toggle()
        try check(studio.streamingQueryKey != queryKey, "Token policy must invalidate options")
        studio.draft = originalDraft
        studio.draft.streaming.status = "available"
        studio.draft.streaming.catalogRevision = "result-only"
        try check(studio.streamingQueryKey == queryKey, "Publishing options must not trigger a query loop")
        studio.draft = originalDraft
        try check(studio.recommendedStreamingSelection == .off,
                  "Physical memory alone cannot recommend an unverified preset")
        let wanDraft = Data(#"{"modelID":"wan2.1-1.3b-qad","modelPaths":{"wan2.1-1.3b-qad":"/models/wan"},"prompt":"keep my prompt","frames":81}"#.utf8)
        let decodedWan = try JSONDecoder().decode(StudioDraft.self, from: wanDraft)
        try check(decodedWan.modelPath == "/models/wan" && decodedWan.frames == 81 &&
                    decodedWan.prompt == "keep my prompt", "Wan draft round trip lost user data")
        studio.draft.modelPaths["flux2-klein-4b"] = "/test/model"
        let output = root.appendingPathComponent("output.png")
        var zStream = StudioDraft()
        zStream.modelID = "z-image-turbo"
        zStream.modelPaths["z-image-turbo"] = "/test/z-image"
        zStream.residency = "streamed"
        zStream.zImageStreamingBudgetGiB = 8
        let zStreamRequest = try zStream.request(output: output)
        try check(zStreamRequest.residency == "streamed" && zStreamRequest.memory_budget_bytes == 8 << 30,
                  "Z-Image streaming budget was not forwarded")
        let restoredStream = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(zStream))
        try check(restoredStream.residency == "streamed" && restoredStream.zImageStreamingBudgetGiB == 8,
                  "Z-Image streaming draft did not persist")
        var publicZ = StudioDraft()
        publicZ.modelID = "z-image-turbo"
        publicZ.modelPaths["z-image-turbo"] = "/test/z-image"
        publicZ.steps = 9
        publicZ.streaming.selection = .tier8
        publicZ.streaming.userSelected = true
        let publicPair = try publicZ.publicStreamingRequest(output: output)
        try check(publicPair.legacy.residency == "resident" &&
                  publicPair.legacy.memory_budget_bytes == nil &&
                  publicPair.v2?.execution.streaming?.target_request_memory_bytes == 8 << 30,
                  "Public selector leaked legacy streamed residency/budget")
        let publicJSON = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(publicPair.v2)) as! [String: Any]
        let publicExecution = publicJSON["execution"] as! [String: Any]
        try check(publicExecution["streaming"] != nil &&
                  publicExecution["residency"] == nil &&
                  publicExecution["memory_budget_bytes"] == nil,
                  "Public V2 intent encoded legacy residency fields")
        let publicOptions = try NativeEngine.streamingOptions(publicPair.v2!)
        try check(publicOptions.query_status == "catalog_empty" &&
                  publicOptions.targets.count == 5 &&
                  publicOptions.targets.allSatisfy { $0.status == "catalog_empty" },
                  "Empty production catalog was not exposed as five unavailable tiers")
        let unopenedOptions = try await NativeEngine.streamingOptions(publicPair.v2!,
            modelURL: URL(fileURLWithPath: "/missing/must-not-be-opened"))
        try check(unopenedOptions.query_status == "catalog_empty",
                  "An empty catalog must not open or load the selected installation")
        var publicFlux4 = publicZ
        publicFlux4.modelID = "flux2-klein-4b"
        publicFlux4.modelPaths[publicFlux4.modelID] = "/test/flux4"
        publicFlux4.steps = 4
        let flux4Pair = try publicFlux4.publicStreamingRequest(output: output)
        try check(publicFlux4.publicStreamingModel && flux4Pair.v2?.model == "flux2-klein-4b" &&
                  flux4Pair.v2?.execution.streaming?.target_request_memory_bytes == 8 << 30 &&
                  flux4Pair.v2?.execution.policy == "gpu" && flux4Pair.v2?.parameters.compile_gpu != true,
                  "Flux 4B public intent must preserve the explicit eager streaming route")
        publicZ.streaming.selection = .off
        let offPair = try publicZ.publicStreamingRequest(output: output)
        try check(offPair.v2 == nil, "Off unexpectedly constructed a V2 selector")
        let offV2 = NativeRequestV2(legacy: offPair.legacy)
        let offJSON = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(offV2)) as! [String: Any]
        try check((offJSON["execution"] as? [String: Any])?["streaming"] == nil,
                  "Off encoded an enabled or zero-byte selector")
        for budget in [6, 8, 10, 12] {
            let migratedLegacy = try JSONDecoder().decode(StudioDraft.self, from: Data(
                "{\"modelID\":\"z-image-turbo\",\"modelPaths\":{\"z-image-turbo\":\"/test/z-image\"},\"residency\":\"streamed\",\"zImageStreamingBudgetGiB\":\(budget)}".utf8))
            let legacyPair = try migratedLegacy.publicStreamingRequest(output: output)
            try check(migratedLegacy.streaming.selection == .off &&
                      !migratedLegacy.streaming.userSelected && legacyPair.v2 == nil &&
                      legacyPair.legacy.residency == "streamed" &&
                      legacyPair.legacy.memory_budget_bytes == UInt64(budget) << 30,
                      "Legacy sampling budget was incorrectly promoted to a public tier")
            let reopenedLegacy = try JSONDecoder().decode(StudioDraft.self,
                from: JSONEncoder().encode(migratedLegacy))
            try check(reopenedLegacy.streaming.selection == .off &&
                      reopenedLegacy.residency == "streamed" &&
                      reopenedLegacy.zImageStreamingBudgetGiB == budget,
                      "Saving the migrated draft lost its original streaming settings")
        }
        var publicLTX = StudioDraft()
        publicLTX.modelID = "ltx-2.5-distilled"
        publicLTX.modelPaths[publicLTX.modelID] = "/test/ltx"
        publicLTX.operation = "video.generate"
        publicLTX.width = 768; publicLTX.height = 448
        publicLTX.steps = 11; publicLTX.frames = 97; publicLTX.fps = 24
        publicLTX.audio = false
        publicLTX.streaming.selection = .tier12
        publicLTX.streaming.userSelected = true
        let ltxOutput = root.appendingPathComponent("public-ltx.mp4")
        let ltxPair = try publicLTX.publicStreamingRequest(output: ltxOutput)
        let capturedLTX = root.appendingPathComponent("public-ltx-request.json")
        let fakeLTX = root.appendingPathComponent("fake-ltx-worker")
        try Data().write(to: ltxOutput)
        try "#!/bin/sh\n/bin/cp \"$3\" \"\(capturedLTX.path)\"\nprintf '{}'\n"
            .write(to: fakeLTX, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700],
                                              ofItemAtPath: fakeLTX.path)
        var rejectedEmptyWorker = false
        do {
            _ = try await LTXWorker.generate(
                model: URL(fileURLWithPath: publicLTX.modelPath),
                request: ltxPair.v2!, outputPath: ltxOutput.path,
                executable: fakeLTX, onEvent: { _ in })
        } catch { rejectedEmptyWorker = true }
        try check(rejectedEmptyWorker && (try Data(contentsOf: ltxOutput)).isEmpty,
                  "Empty worker result must fail and preserve the existing output")
        let capturedLTXJSON = try JSONSerialization.jsonObject(
            with: Data(contentsOf: capturedLTX)) as! [String: Any]
        let capturedExecution = capturedLTXJSON["execution"] as! [String: Any]
        let capturedSelector = capturedExecution["streaming"] as! [String: Any]
        let stagedOutput = (capturedLTXJSON["outputs"] as! [[String: Any]])[0]["path"] as! String
        try check(stagedOutput != ltxOutput.path && stagedOutput.contains(".tc-ltx-staging-"),
                  "Worker must write into a fresh request-scoped staging directory")
        let capturedTarget = (capturedSelector["target_request_memory_bytes"] as? NSNumber)?.uint64Value
        try check(capturedLTXJSON["schema_version"] as? Int == 2 &&
                  capturedTarget == 12 << 30 &&
                  capturedExecution["residency"] == nil,
                  "LTX worker did not receive a worker-local V2 public intent")
        let adapter = root.appendingPathComponent("stream-conflict-lora.safetensors")
        try Data([0]).write(to: adapter)
        var conflictDraft = restoredStream
        conflictDraft.loras = [StudioLoRA(path: adapter.path, strength: 0.8)]
        for prompt in ["a fox", "日落时分的湖泊"] {
            conflictDraft.prompt = prompt
            try check(conflictDraft.zImageStreamingConflict()?.contains("LoRA") == true,
                      "Streaming conflict incorrectly depends on the prompt")
            do {
                _ = try conflictDraft.request(output: output)
                throw NativeFailure(message: "Streaming accepted active LoRA")
            } catch {
                try check(error.localizedDescription == conflictDraft.zImageStreamingConflict(),
                          "UI and request validation disagree on the streaming conflict")
            }
        }
        let conflictDirectory = root.appendingPathComponent("stream-conflict")
        let conflictStudio = StudioState(directory: conflictDirectory)
        conflictStudio.draft = conflictDraft
        conflictStudio.save()
        let restoredConflict = StudioState(directory: conflictDirectory)
        try check(restoredConflict.draft.zImageStreamingConflict()?.contains("LoRA") == true,
                  "Saved invalid streaming settings hid the recovery action")
        restoredConflict.useZImageResidentLoading()
        let recovered = try restoredConflict.draft.request(output: output)
        try check(recovered.residency == "resident" && recovered.memory_budget_bytes == nil &&
                  recovered.loras?.count == 1 && restoredConflict.draft.loras == conflictDraft.loras &&
                  recovered.prompt == conflictDraft.prompt && recovered.seed == 42,
                  "Streaming recovery lost LoRA, prompt or generation parameters")
        try check(StudioState(directory: conflictDirectory).draft.residency == "resident",
                  "Streaming recovery was not persisted")
        restoredConflict.draft.streaming.selection = .tier16
        try check(restoredConflict.draft.zImageStreamingConflict()?.contains("LoRA") == true,
                  "Public streaming conflict did not expose recovery")
        restoredConflict.useZImageResidentLoading()
        let recoveredPublic = try restoredConflict.draft.publicStreamingRequest(output: output)
        try check(recoveredPublic.v2 == nil && recoveredPublic.legacy.residency == "resident" &&
                  recoveredPublic.legacy.loras?.count == 1 &&
                  StudioState(directory: conflictDirectory).draft.streaming.selection == .off,
                  "Recovery left the public streaming selector enabled or dropped LoRA")
        conflictDraft.loras[0].enabled = false
        try check(conflictDraft.zImageStreamingConflict() == nil,
                  "Disabled LoRA blocked GPU streaming")
        conflictDraft.acceleration = StudioAcceleration(policy: "gpu_ane")
        try check(conflictDraft.zImageStreamingConflict(systemJSON: "{}")?.contains("ANE") == true,
                  "Unsupported ANE streaming did not explain the conflict")
        conflictDraft.acceleration = StudioAcceleration(policy: "auto")
        try check(conflictDraft.zImageStreamingConflict()?.contains("加速配置") == true,
                  "Automatic streaming policy did not explain the conflict")
        conflictDraft.acceleration = nil; conflictDraft.profilePath = "/test/profile.json"
        try check(conflictDraft.zImageStreamingConflict()?.contains("加速配置") == true,
                  "Legacy profile streaming did not explain the conflict")
        zStream.acceleration = StudioAcceleration(policy: "gpu_ane")
        try rejects { _ = try zStream.request(output: output) }
        zStream.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/test/z-image/compiled.json")
        if AccelerationDiscovery.optimizationEnabled("z_image_suffix_streaming") {
            let zHybridStreamRequest = try zStream.request(output: output)
            try check(zHybridStreamRequest.execution == "gpu_ane" &&
                    zHybridStreamRequest.ane_manifest == "/test/z-image/compiled.json" &&
                    zHybridStreamRequest.allow_approximation == true &&
                    zHybridStreamRequest.memory_budget_bytes == 8 << 30,
                      "Explicit Z-Image suffix streaming configuration was not forwarded")
        } else {
            try rejects { _ = try zStream.request(output: output) }
        }
        try check(!AccelerationDiscovery.optimizationEnabled("z_image_suffix_streaming", systemJSON: "{}"),
                  "Old engine without device policy enabled M5 optimization")
        try check(!AccelerationDiscovery.optimizationEnabled("unknown"), "Unknown optimization was enabled")
        try check(!AccelerationDiscovery.optimizationEnabled("z_image_suffix_streaming",
            systemJSON: "{\"optimization_profile\":{\"id\":\"legacy\",\"z_image_suffix_streaming\":false}}"),
                  "Legacy device policy enabled hybrid streaming")
        zStream.acceleration = StudioAcceleration(policy: "auto")
        try rejects { _ = try zStream.request(output: output) }
        zStream.acceleration = nil
        zStream.zImageStreamingBudgetGiB = -1
        try rejects { _ = try zStream.request(output: output) }
        let textOnly = StudioModel(id: "flux2-klein-4b", name: "Text only fixture", executor: true, output: "image", operations: ["image.generate"], default_steps: 4, default_frames: 1, default_width: 512, default_height: 512)
        try studio.draft.validate(models: [textOnly])
        let restricted = StudioState(directory: root.appendingPathComponent("restricted"), models: [textOnly])
        restricted.draft.operation = "image.edit"
        restricted.changeModel(textOnly.id)
        try check(restricted.draft.operation == "image.generate" && !restricted.supportsImageInputs, "Model switch did not normalize unsupported operation")
        restricted.changeOperation("image.edit")
        try check(restricted.draft.operation == "image.generate", "Unsupported operation entered through UI")
        await restricted.addFiles([root.appendingPathComponent("unused.png")])
        try check(restricted.draft.assets.isEmpty && restricted.message?.contains("未开放") == true, "Text-only import not rejected")
        var unsupported = studio.draft; unsupported.operation = "image.edit"
        try rejects { try unsupported.validate(models: [textOnly]) }
        unsupported.modelID = "missing-model"
        try rejects { try unsupported.validate() }
        try check(try studio.draft.request(output: output).seed == 42, "Default seed changed")
        try check(try studio.draft.request(output: output).execution == "gpu", "New drafts must default to GPU only")
        let legacyAdapter = try JSONDecoder().decode(StudioLoRA.self, from: Data("{\"path\":\"/missing-adapter\"}".utf8))
        try check(legacyAdapter.enabled && legacyAdapter.strength == 1.0, "Legacy LoRA defaults changed")
        studio.draft.loras = [StudioLoRA(path: "/missing-adapter", strength: 0.6, enabled: false)]
        try check(try studio.draft.request(output: output).loras == nil, "Disabled LoRA was validated or forwarded")
        let roundTrip = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(studio.draft))
        try check(!roundTrip.loras[0].enabled && roundTrip.loras[0].strength == 0.6, "LoRA switch or strength did not persist")
        studio.draft.loras = []
        studio.setANEEnabled(true)
        try check(studio.draft.usesANE, "ANE toggle did not enable hybrid")
        studio.setANEEnabled(false)
        try check(try studio.draft.request(output: output).execution == "gpu", "ANE toggle did not restore GPU")
        var oldDraftJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(studio.draft)) as! [String: Any]
        oldDraftJSON.removeValue(forKey: "acceleration")
        let oldDraft = try JSONDecoder().decode(StudioDraft.self, from: JSONSerialization.data(withJSONObject: oldDraftJSON))
        try check(oldDraft.acceleration == nil, "Older saved draft must remain readable")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/test/compiled/manifest.json", sourceManifest: "")
        let hybridRequest = try studio.draft.request(output: output)
        try check(hybridRequest.execution == "gpu_ane" && hybridRequest.allow_approximation == true && hybridRequest.ane_manifest != nil, "Hybrid mode not forwarded")
        studio.draft.acceleration = StudioAcceleration(policy: "auto")
        let autoRequest = try studio.draft.request(output: output)
        try check(autoRequest.execution == "auto" && autoRequest.allow_approximation == true && autoRequest.ane_manifest == nil, "Automatic missing-partition request must allow GPU fallback")
        let fixture = root.appendingPathComponent("model")
        let weight = fixture.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors")
        try FileManager.default.createDirectory(at: weight.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: weight)
        let compiled = fixture.appendingPathComponent("coreml")
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        var artifacts: [String: [String: String]] = [:]
        for i in 0..<20 {
            let name = "block\(i).mlmodelc"
            try FileManager.default.createDirectory(at: compiled.appendingPathComponent(name), withIntermediateDirectories: true)
            artifacts[String(i)] = ["int8_pc": name]
        }
        var manifest: [String: Any] = ["schema_version": 2, "shape": ["K": 3072, "N": 3072, "buckets": [1088]], "source": ["checkpoint": weight.path, "checkpoint_bytes": 3], "artifacts": artifacts]
        let manifestFile = compiled.appendingPathComponent("manifest.json")
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path, cache: compiled)?.rows == 1088, "Compatible local partition not found")
        var cachedDraft = studio.draft
        cachedDraft.modelPaths[cachedDraft.modelID] = fixture.path
        cachedDraft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: manifestFile.path)
        let resolvedCache = try await inventoryStore.resolveAcceleration(cachedDraft)
        try check(resolvedCache.acceleration?.manifest == manifestFile.path && inventoryStore.accelerationStatus?.contains("未重新编译") == true,
                  "An explicitly selected compiled partition outside the App cache was not reused")
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path, cache: compiled, minimumRows: 4097) == nil, "1024 task accepted undersized partition")
        var zArtifacts: [String: [String: String]] = [:]
        for i in 0..<32 {
            let name = "z-block\(i).mlmodelc"
            try FileManager.default.createDirectory(at: compiled.appendingPathComponent(name), withIntermediateDirectories: true)
            zArtifacts[String(i)] = ["int8_pc": name]
        }
        func zPartition(_ name: String, _ buckets: [Int], end: Int = 4096) throws -> String {
            let file = compiled.appendingPathComponent(name + ".json")
            let value: [String: Any] = [
                "schema_version": 2,
                "shape": ["K": 3840, "N": 3840, "buckets": buckets, "input_mode": "enumerated",
                          "mlp_width": 10240, "ane_mlp_start": 0, "ane_mlp_end": end],
                "source": ["checkpoint": weight.path, "checkpoint_bytes": 3], "artifacts": zArtifacts
            ]
            try JSONSerialization.data(withJSONObject: value).write(to: file)
            return file.path
        }
        let zSmall = try zPartition("z-small", [1056, 1120, 1536])
        let zLarge = try zPartition("z-large", [4128, 4192, 5120])
        let zSameRows = try zPartition("z-same-rows", [1120])
        let zOtherSplit = try zPartition("z-other-split", [1088, 1120], end: 6144)
        func zMatch(_ preferred: String, _ rows: Int, _ known: [String]) -> AccelerationDiscovery.Match? {
            AccelerationDiscovery.find(modelPath: fixture.path, preferred: preferred, cache: compiled,
                minimumRows: rows, modelID: "z-image-turbo", knownManifests: known,
                preferSmallestRows: true)
        }
        try check(zMatch(zLarge, 1120, [zSmall])?.manifest == zSmall,
                  "Switching from 1024 to 512 retained the oversized ANE partition")
        try check(zMatch(zSmall, 4192, [zLarge])?.manifest == zLarge,
                  "Switching from 512 to 1024 failed to select sufficient ANE capacity")
        try check(zMatch(zSameRows, 1120, [zSmall])?.manifest == zSameRows,
                  "Equal-row selection needlessly replaced the preferred partition")
        try check(zMatch(zLarge, 1088, [zSmall, zOtherSplit])?.manifest == zSmall,
                  "Row minimization changed the selected MLP channel split")
        try check(zMatch(zLarge, 1120, [])?.manifest == zLarge,
                  "An oversized partition must remain usable when no smaller equivalent exists")
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: zLarge, cache: compiled,
                    minimumRows: 1120, modelID: "z-image-turbo", knownManifests: [zSmall])?.manifest == zLarge,
                  "Explicit discovery preference changed without opting into row minimization")
        let discoveryLoRAFile = root.appendingPathComponent("discovery-lora.safetensors")
        try Data([7, 8]).write(to: discoveryLoRAFile)
        let discoveryLoRA = StudioLoRA(path: discoveryLoRAFile.path, strength: 0.6)
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path,
                                             cache: compiled, loras: [discoveryLoRA]) == nil,
                  "Base partition was reused for an active LoRA")
        manifest["source"] = [
            "checkpoint": weight.path,
            "checkpoint_bytes": 3,
            "loras": [["path": discoveryLoRAFile.path, "bytes": 2,
                        "sha256": String(repeating: "a", count: 64),
                        "role": "transformer", "strength": 0.6]]
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path,
                                             cache: compiled, loras: [discoveryLoRA])?.rows == 1088,
                  "LoRA-bound partition was not discovered")
        try check(AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M4 Max", memory: 64 * 1024 * 1024 * 1024, mlpWidth: 9216, start: 0, end: 6144), "M4 Max validated prefix was rejected")
        try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M4 Max", memory: 64 * 1024 * 1024 * 1024, mlpWidth: 9216, start: 0, end: 9216), "M4 Max accepted the unvalidated full MLP partition")
        try check(AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M4 Max", memory: 64 * 1024 * 1024 * 1024, mlpWidth: 10240, start: 0, end: 4096, modelID: "z-image-turbo"), "Validated Z-Image M4 Max prefix was rejected")
        try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M4 Max", memory: 64 * 1024 * 1024 * 1024, mlpWidth: 10240, start: 0, end: 5120, modelID: "z-image-turbo"), "Unvalidated Z-Image prefix bypassed the automatic policy")
        try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M4 Pro", memory: 48 * 1024 * 1024 * 1024, mlpWidth: 10240, start: 0, end: 4096, modelID: "z-image-turbo"), "M4 Max Z-Image policy leaked to M4 Pro")
        try check(AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M5 Pro", memory: 24 << 30, mlpWidth: 9216, start: 0, end: 6144, bucket: 1088), "Measured M5 Pro partition was rejected")
        for end in [3072, 9216] {
            try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M5 Pro", memory: 24 << 30, mlpWidth: 9216, start: 0, end: end, bucket: 1088), "Unqualified M5 partition became automatic")
        }
        for gpu in ["Apple M5", "Apple M5 Max"] {
            try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: gpu, memory: 24 << 30, mlpWidth: 9216, start: 0, end: 6144, bucket: 1088), "M5 Pro policy leaked to another chip")
        }
        try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M5 Pro", memory: 48 << 30, mlpWidth: 9216, start: 0, end: 6144), "M5 Pro policy leaked to another memory configuration")
        try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M5 Pro", memory: 24 << 30, mlpWidth: 9216, start: 0, end: 6144, bucket: 4160), "Unmeasured M5 1024 bucket was accepted")
        try check(!AccelerationDiscovery.automaticPolicyMatches(gpu: "Apple M5 Pro", memory: 24 << 30, mlpWidth: 10240, start: 0, end: 4096, modelID: "z-image-turbo"), "M5 FLUX policy leaked to Z-Image")
        // Offline compilation and the model library need not use the App cache.
        manifest["source"] = ["checkpoint": weight.path, "checkpoint_bytes": 3]
        manifest["shape"] = ["K": 3072, "N": 3072, "buckets": [1088],
                             "mlp_width": 9216, "ane_mlp_start": 0, "ane_mlp_end": 6144]
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        let hardware = try JSONSerialization.jsonObject(with: Data(NativeEngine.system().utf8)) as! [String: Any]
        let automaticExpected = AccelerationDiscovery.automaticPolicyMatches(
            gpu: hardware["gpu"] as? String ?? "", memory: (hardware["physical_memory_bytes"] as? NSNumber)?.uint64Value ?? 0,
            mlpWidth: 9216, start: 0, end: 6144, bucket: 1088) &&
            AccelerationDiscovery.optimizationEnabled("external_automatic_partitions")
        let emptyCache = root.appendingPathComponent("external-auto-cache")
        let externalAuto = AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path,
            cache: emptyCache, minimumRows: 1044, requiredRows: 1088, enforceAutomaticPolicy: true)
        try check((externalAuto?.manifest == manifestFile.path) == automaticExpected,
                  "External preferred partition did not follow the automatic hardware gate")
        let registry = try LibraryStore()
        let registered = try registry.registerANE(modelID: "flux2-klein-4b", manifest: manifestFile)
        let registeredAuto = AccelerationDiscovery.find(modelPath: fixture.path,
            cache: emptyCache, minimumRows: 1044, requiredRows: 1088, enforceAutomaticPolicy: true)
        try check((registeredAuto?.manifest == manifestFile.path) == automaticExpected,
                  "Registered external partition was not checked for automatic discovery")
        try check(AccelerationDiscovery.find(modelPath: fixture.path, cache: emptyCache,
            minimumRows: 1044, requiredRows: 1088, enforceAutomaticPolicy: true,
            loras: [discoveryLoRA]) == nil, "Registry bypassed the automatic LoRA gate")
        try registry.unregisterANE(id: registered.id)
        manifest["source"] = ["checkpoint": weight.path, "checkpoint_bytes": 4]
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path, cache: compiled) == nil, "Wrong checkpoint accepted")
        manifest["source"] = ["checkpoint": weight.path, "checkpoint_bytes": 3]
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        try FileManager.default.removeItem(at: compiled.appendingPathComponent("block19.mlmodelc"))
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path, cache: compiled) == nil, "Incomplete partition accepted")
        var resourceDraft = studio.draft
        resourceDraft.acceleration = StudioAcceleration(exportProfile: "/unavailable/build-profile.json")
        resourceDraft.profilePath = "/unavailable/inference-profile.json"
        try check(resourceDraft.coreMLResourceRequest("inventory")["profile"] == nil, "Inventory opened an unrelated build profile")
        resourceDraft.acceleration?.exportPython = "/unavailable/python"
        resourceDraft.acceleration?.exportPythonPath = "/unavailable/packages"
        for action in ["inventory", "compile", "export"] {
            let resource = resourceDraft.coreMLResourceRequest(action)
            try check(resource["profile"] == nil && resource["python"] == nil && resource["python_path"] == nil, "Runtime resources leaked developer toolchain settings")
        }
        studio.draft.acceleration = nil
        studio.draft.seedText = "123"
        try check(try studio.draft.request(output: output).seed == 123, "Fixed seed ignored")
        for invalid in ["-1", "2147483648", "1.5", "", "random"] {
            studio.draft.seedText = invalid
            try rejects { _ = try studio.draft.request(output: output) }
        }
        studio.draft.seedText = "42"; studio.draft.randomSeed = true
        var randomCalls = 0
        let snapshot = try studio.draft.request(output: output, random: { randomCalls += 1; return 765 })
        try check(randomCalls == 1 && snapshot.seed == 765 && snapshot.frames == 1, "Random seed/output not frozen")
        studio.draft.seedText = "100"
        try check(snapshot.seed == 765, "Draft mutated snapshot")
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 32, pixelsHigh: 24, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        let png = bitmap.representation(using: .png, properties: [:])!
        let input = root.appendingPathComponent("source.png")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try png.write(to: input)
        try await verifyEditingWorkflows(root: root.appendingPathComponent("editing-workflows"), source: input, png: png)
        await studio.addFiles([input, input])
        try check(studio.draft.assets.count == 2 && studio.draft.assets[0].id != studio.draft.assets[1].id, "Duplicate source identity collided")
        let firstID = studio.draft.assets[0].id
        studio.changeOperation("image.edit")
        let edit = try studio.draft.request(output: output)
        try check(edit.inputs?.count == 2 && edit.inputs?.allSatisfy { $0.role == "reference" } == true, "Reference mapping failed")
        studio.move(firstID, offset: 1)
        try check(studio.draft.assets.last?.id == firstID, "Reference reordering failed")
        studio.undoAssetChange()
        try check(studio.draft.assets.first?.id == firstID, "Undo order failed")
        studio.changeOperation("image.transform")
        try check(try studio.draft.request(output: output).inputs?.count == 1, "Transform did not select exactly one input")
        studio.changeOperation("image.generate")
        try check(studio.draft.assets.count == 2 && studio.draft.activeAssets.isEmpty, "Operation switch lost unused inputs")
        studio.remove(firstID)
        try check(FileManager.default.fileExists(atPath: edit.inputs![0].path), "Removing draft input deleted in-flight asset")
        studio.undoAssetChange()
        await studio.addFiles(Array(repeating: input, count: 7))
        try check(studio.draft.assets.count == 2, "Overflow silently imported/truncated inputs")
        // One pasteboard item has multiple representations; it must create one image.
        let board = NSPasteboard.withUniqueName(); defer { board.releaseGlobally() }
        board.clearContents()
        board.declareTypes([.png, .tiff], owner: nil)
        if board.setData(png, forType: .png), board.setData(bitmap.tiffRepresentation!, forType: .tiff),
           !(board.types ?? []).isEmpty {
            await studio.pasteImage(from: board)
            let clipboardMessage = studio.message ?? "no message"
            try check(studio.draft.assets.count == 3,
                      "Clipboard image import count was \(studio.draft.assets.count): \(clipboardMessage)")
        } else {
            print("SKIP: pasteboard service unavailable in this session")
        }
        let countBeforeFailure = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("inputs").path).count
        await studio.addFiles([input, root.appendingPathComponent("missing.png")])
        let expectedAssetCount = (board.types ?? []).isEmpty ? 2 : 3
        try check(studio.draft.assets.count == expectedAssetCount, "Partial import mutated draft")
        try check(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("inputs").path).count == countBeforeFailure, "Partial import leaked staged file")
        studio.save()
        let restored = StudioState(directory: root)
        try check(restored.draft.assets.map(\.id) == studio.draft.assets.map(\.id), "Draft identity/order did not persist")
        var telemetry = StepTelemetry()
        func event(_ sequence: Int, _ completed: Int, _ seconds: Double, _ phase: String = "denoise") -> NativeEvent { NativeEvent(sequence: sequence, phase: phase, completed: completed, total: 6, elapsed_seconds: seconds) }
        try check(telemetry.observe(event(1, 1, 10)) == nil, "Transform offset counted as completed sample")
        try check(telemetry.observe(event(2, 2, 12)) == nil, "Single sample shown as reliable speed")
        try check(telemetry.firstDenoiseStepSeconds == nil, "Missing step-zero boundary fabricated a first-step timing")
        _ = telemetry.observe(event(3, 2, 13)); _ = telemetry.observe(event(4, 30, 13.5, "block"))
        try check(telemetry.observe(event(5, 3, 15)) == 2.5, "Step duration includes wrong boundaries")
        try check(telemetry.observe(event(2, 4, 50)) == 2.5, "Out-of-order telemetry altered speed")
        try check(telemetry.firstDenoiseStepSeconds == nil, "Later progress fabricated the unobserved first interval")

        // A slow initial sampling step must age out of the rolling window,
        // while its separate duration remains available in result history.
        var editTelemetry = StepTelemetry()
        _ = editTelemetry.observe(event(1, 0, 10, "load_qwen21_transformer"))
        _ = editTelemetry.observe(event(2, 1, 20, "qwen21_text_encode"))
        try check(editTelemetry.secondsPerStep == nil && editTelemetry.firstDenoiseStepSeconds == nil,
                  "Preparation phases created sampling timing")
        _ = editTelemetry.observe(event(3, 0, 30))
        try check(editTelemetry.firstDenoiseStepSeconds == nil, "Initial denoise boundary counted as a completed step")
        _ = editTelemetry.observe(event(4, 1, 42))
        try check(editTelemetry.firstDenoiseStepSeconds == 12 && editTelemetry.secondsPerStep == nil,
                  "First sampling duration included preparation or appeared as a reliable average")
        _ = editTelemetry.observe(event(5, 1, 43))
        _ = editTelemetry.observe(event(6, 31, 43.5, "transformer_block"))
        try check(editTelemetry.observe(event(7, 2, 44)) == 7,
                  "Repeated boundaries or block detail altered the sampling window")
        for completed in 3...5 {
            _ = editTelemetry.observe(event(completed + 5, completed, 40 + Double(completed) * 2))
        }
        try check(editTelemetry.secondsPerStep == 4, "Five-step window dropped the first step too early")
        try check(editTelemetry.observe(event(11, 6, 52)) == 2 && editTelemetry.firstDenoiseStepSeconds == 12,
                  "Six-step result did not retain its first duration separately from the last five steps")

        var timedJob = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "timing history fixture", output: root.appendingPathComponent("timing.png").path),
                                 state: "succeeded", phase: "complete", completed: 6, total: 6, elapsed: 60)
        timedJob.secondsPerStep = editTelemetry.secondsPerStep
        timedJob.firstDenoiseStepSeconds = editTelemetry.firstDenoiseStepSeconds
        let timingData = try JSONEncoder().encode(timedJob)
        let decodedTiming = try JSONDecoder().decode(NativeJob.self, from: timingData)
        try check(decodedTiming.firstDenoiseStepSeconds == 12 && decodedTiming.secondsPerStep == 2,
                  "Sampling history lost the distinct first-step and rolling timings")
        var legacyTiming = try JSONSerialization.jsonObject(with: timingData) as! [String: Any]
        legacyTiming.removeValue(forKey: "firstDenoiseStepSeconds")
        let decodedLegacy = try JSONDecoder().decode(NativeJob.self, from: JSONSerialization.data(withJSONObject: legacyTiming))
        try check(decodedLegacy.firstDenoiseStepSeconds == nil && decodedLegacy.secondsPerStep == 2,
                  "Older job history without a first-step field is no longer compatible")
        _ = editTelemetry.observe(event(12, 0, 54))
        try check(editTelemetry.firstDenoiseStepSeconds == nil && editTelemetry.secondsPerStep == nil,
                  "A restarted sampling sequence retained stale timings")
        _ = editTelemetry.observe(event(13, 1, 57))
        try check(editTelemetry.firstDenoiseStepSeconds == 3, "Restarted sampling did not record its own first interval")
        for model in studio.models { studio.draft.modelPaths[model.id] = "/test/\(model.id)" }
        studio.selectModel("minimax-h3-turbo")
        try check(studio.draft.operation == "video.generate" && studio.draft.frames == 22 && studio.draft.fps == 24 && studio.draft.audio,
                  "H3 defaults were not applied")
        studio.draft.assets = Array(studio.draft.assets.prefix(2))
        studio.changeOperation("video.keyframes")
        studio.draft.initImageID = studio.draft.assets.first?.id
        let h3 = try studio.draft.request(output: root.appendingPathComponent("h3.mp4"))
        try check(h3.inputs?.map(\.role) == ["first_frame", "last_frame"], "H3 keyframe roles were not mapped")
        studio.selectModel("ltx-2.5-distilled")
        try check(studio.draft.operation == "video.generate", "LTX did not select its public executor operation")
        let ltxText = try studio.draft.request(output: root.appendingPathComponent("ltx-text.mp4"))
        try check(ltxText.execution == "gpu" && ltxText.allow_approximation != true &&
                    LTXWorker.accepts(ltxText), "LTX must default to exact GPU in a disposable worker")
        studio.changeOperation("video.image")
        try check(studio.draft.operation == "video.image", "LTX I2V operation was not selectable")
        let ltx = try studio.draft.request(output: root.appendingPathComponent("ltx.mp4"))
        try check(ltx.width == 704 && ltx.height == 448 && ltx.frames == 97 &&
                    ltx.steps == 11 && ltx.inputs?.map(\.role) == ["first_frame"],
                  "LTX descriptor defaults or public image-to-video mapping changed")
        studio.draft.residency = "streamed"
        try rejects { try studio.draft.validate() }
        studio.draft.residency = "component_staged"
        try check(LTXWorker.accepts(ltx), "LTX I2V must also use the disposable worker")
        studio.draft.audio = true
        let ltxAV = try studio.draft.request(
            output: root.appendingPathComponent("ltx-ax-audio.mp4"))
        try check(ltxAV.audio == true && ltxAV.inputs?.map(\.role) == ["first_frame"],
                  "LTX image-to-video audio request was not forwarded")
        studio.draft.ltxBackend = "c_metal"
        studio.draft.ltxVideoAttentionBatch = true
        let ltxExactBatch = try studio.draft.request(
            output: root.appendingPathComponent("ltx-exact-batch.mp4"))
        try check(ltxExactBatch.ltx_video_attention_batch == true &&
                    ltxExactBatch.allow_approximation != true,
                  "LTX exact Video attention batch was not forwarded")
        studio.draft.ltxAccelerationMode = "fast_approx"
        let ltxFast = try studio.draft.request(
            output: root.appendingPathComponent("ltx-fast-approx.mp4"))
        try check(ltxFast.allow_approximation == true &&
                    ltxFast.ltx_sol_stage2 == true &&
                    ltxFast.ltx_sol_tau == 1.0 &&
                    ltxFast.ltx_sol_dense_edge_blocks == 1 &&
                    ltxFast.ltx_sol_dense_edge_steps == 0 &&
                    ltxFast.ltx_stage2_text_rows == 256 &&
                    ltxFast.ltx_video_attention_batch == false &&
                    ltxFast.inputs?.map(\.role) == ["first_frame"],
                  "LTX optional Sol/text-pruned I2V mode was not forwarded")
        let ltxDescriptor = studio.models.first { $0.id == "ltx-2.5-distilled" }!
        for backend in ["auto", "c_metal"] {
            for mode in ["sol", "fast_approx"] {
                var bounded = studio.draft
                bounded.operation = "video.generate"; bounded.assets = []; bounded.initImageID = nil
                bounded.audio = false; bounded.ltxBackend = backend; bounded.ltxAccelerationMode = mode
                try bounded.validate(model: ltxDescriptor)
                // The 8n+1 value reaches the old Stage-2 multiplication. Large
                // aligned dimensions also passed its old model-specific guards.
                var hugeFrames = bounded
                hugeFrames.width = 1280; hugeFrames.height = 704
                hugeFrames.frames = 1_000_000_000_000_000_001
                try rejects { try hugeFrames.validate(model: ltxDescriptor) }
                for value in [Int.max - 63, Int.min] {
                    var hugeWidth = bounded; hugeWidth.width = value
                    try rejects { try hugeWidth.validate(model: ltxDescriptor) }
                    var hugeHeight = bounded; hugeHeight.height = value
                    try rejects { try hugeHeight.validate(model: ltxDescriptor) }
                }
                var negativeFrames = bounded; negativeFrames.frames = Int.min
                try rejects { try negativeFrames.validate(model: ltxDescriptor) }
                bounded.width = 64; bounded.height = 64; bounded.frames = 361
                try bounded.validate(model: ltxDescriptor)
            }
        }
        studio.draft.audio = false
        studio.selectModel("wan2.1-1.3b-qad")
        try check(studio.draft.loraStrategy == "auto", "Model switch did not reset LoRA strategy")
        let lora = root.appendingPathComponent("adapter.safetensors"); try Data([9]).write(to: lora)
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        let wan = try studio.draft.request(output: root.appendingPathComponent("wan.mp4"))
        try check(wan.width == 832 && wan.height == 480 && wan.frames == 81 && wan.fps == 16 && wan.execution == "gpu" && wan.loras?.count == 1,
                  "Wan defaults or separate LoRA forwarding changed")
        studio.selectModel("flux2-klein-4b")
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        studio.draft.loraStrategy = "in_memory_merge"
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/test/compiled/manifest.json")
        let fluxLoRA = try studio.draft.request(output: root.appendingPathComponent("flux-lora.png"))
        try check(fluxLoRA.execution == "gpu" && fluxLoRA.ane_manifest == nil &&
                    fluxLoRA.loras?.count == 1 && fluxLoRA.lora_strategy == "in_memory_merge",
                  "FLUX separate LoRA request did not safely avoid the base ANE artifact")
        let loraManifest = root.appendingPathComponent("lora-aware-manifest.json")
        let loraIdentity: [String: Any] = [
            "path": lora.path, "bytes": 1,
            "sha256": String(repeating: "0", count: 64),
            "role": "transformer", "strength": 0.8
        ]
        try JSONSerialization.data(withJSONObject: [
            "schema_version": 2, "source": ["loras": [loraIdentity]]
        ]).write(to: loraManifest)
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: loraManifest.path)
        let fluxLoRAHybrid = try studio.draft.request(output: root.appendingPathComponent("flux-lora-hybrid.png"))
        try check(fluxLoRAHybrid.execution == "gpu_ane" &&
                    fluxLoRAHybrid.ane_manifest == loraManifest.path,
                  "FLUX LoRA-bound ANE artifact was not forwarded")
        studio.selectModel("flux2-klein-9b")
        studio.draft.loras = []
        let flux9 = try studio.draft.request(output: root.appendingPathComponent("flux9.png"))
        try check(flux9.model == "flux2-klein-9b" && flux9.operation == "image.generate" && flux9.frames == 1,
                  "FLUX 9B App selection changed")
        studio.selectModel("z-image-turbo")
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.7)]
        studio.draft.loraStrategy = "in_memory_merge"
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
        let zImage = try studio.draft.request(output: root.appendingPathComponent("z-image.png"))
        try check(zImage.model == "z-image-turbo" && zImage.operation == "image.generate" &&
                    zImage.width == 512 && zImage.height == 512 && zImage.steps == 8 &&
                    zImage.frames == 1 && zImage.audio == false && zImage.execution == "gpu" &&
                    zImage.loras?.first?.role == "transformer",
                  "Z-Image App defaults, GPU fail-closed policy or separate LoRA forwarding changed")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: loraManifest.path)
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        let zImageLoRAHybrid = try studio.draft.request(output: root.appendingPathComponent("z-image-lora-hybrid.png"))
        try check(zImageLoRAHybrid.execution == "gpu_ane" &&
                    zImageLoRAHybrid.ane_manifest == loraManifest.path,
                  "Z-Image LoRA-bound ANE artifact was not forwarded")
        try check(studio.draft.executionDeviceLabel == "GPU + Core ML" && studio.draft.aneConfigurationNotice == nil,
                  "Bound frozen LoRA partition was incorrectly displayed as GPU")
        var runtimeZ = studio.draft
        runtimeZ.loraStrategy = "inference_time"
        let runtimeLoRAStore = NativeJobStore(directory: root.appendingPathComponent("runtime-lora-routing"))
        let resolvedRuntimeZ = try await runtimeLoRAStore.resolveAcceleration(runtimeZ)
        try check(resolvedRuntimeZ.acceleration?.manifest == runtimeZ.acceleration?.manifest &&
                  !runtimeLoRAStore.resolvingAcceleration && runtimeLoRAStore.accelerationStatus?.contains("GPU") == true,
                  "GPU runtime LoRA tried to resolve an unused frozen ANE partition")
        let runtimeZRequest = try runtimeZ.request(output: root.appendingPathComponent("unused-runtime-z.png"))
        try check(runtimeZRequest.execution == "gpu" && runtimeZRequest.ane_manifest == nil &&
                  runtimeZ.executionDeviceLabel == "GPU" && runtimeZ.aneConfigurationNotice?.contains("运行时加载") == true &&
                  runtimeZ.accelerationHint.hasPrefix("GPU"),
                  "Runtime LoRA display disagreed with the GPU request")
        runtimeZ.loras[0].enabled = false
        try check(runtimeZ.executionDeviceLabel == "GPU + Core ML" && runtimeZ.aneConfigurationNotice == nil,
                  "Disabled LoRA left a stale GPU notice")
        let preparedZ = try studio.preparationRequest(modelID: "z-image-turbo", output: output)
        try check(preparedZ.loras?.first?.strength == 0.8 && preparedZ.ane_manifest == loraManifest.path && preparedZ.width == 512,
                  "Loading the current model discarded LoRA, dimensions or acceleration")
        for steps in [1, 4, 8, 9, 20, 50] {
            studio.draft.steps = steps
            let configurable = try studio.preparationRequest(modelID: "z-image-turbo", output: output)
            try check(configurable.steps == steps, "Z-Image custom steps were discarded")
            studio.changeModel("z-image-turbo")
            try check(studio.draft.steps == steps, "Reselecting the current model reset custom steps")
        }
        studio.save()
        let restoredSteps = StudioState(directory: root)
        try check(restoredSteps.draft.steps == 50, "Custom steps did not persist")
        studio.draft.steps = 9
        var invalidZ = studio.draft; invalidZ.steps = 0
        try rejects { try invalidZ.validate() }
        invalidZ.steps = 51
        try rejects { try invalidZ.validate() }
        invalidZ.steps = 9; invalidZ.loras[0].role = "text_encoder"
        try rejects { try invalidZ.validate() }
        let creationStudio = StudioState(directory: root.appendingPathComponent("creation-first"))
        try check(creationStudio.creationOperations.contains("image.generate"), "Image entry missing without installed models")
        creationStudio.changeCreationKind("video")
        try check(creationStudio.creationKind == "video" && creationStudio.draft.operation == "video.generate", "Video entry did not select a compatible executor")
        try check(creationStudio.creationModels.allSatisfy { $0.availableOperations.contains { $0.hasPrefix("video.") } }, "Image model leaked into video choices")
        creationStudio.changeCreationKind("image")
        creationStudio.changeOperation("image.edit")
        try check(creationStudio.draft.operation == "image.edit" && creationStudio.models.first { $0.id == creationStudio.draft.modelID }!.supports("image.edit"), "Operation-first selection failed")
        creationStudio.changeModel("flux2-klein-9b")
        try check(creationStudio.draft.operation == "image.edit", "Compatible model switch discarded operation")
        let previousCreation = creationStudio.draft
        creationStudio.changeOperation("unsupported.operation")
        try check(creationStudio.draft.modelID == previousCreation.modelID && creationStudio.draft.operation == previousCreation.operation, "Unsupported operation changed the draft")
        let comfy = root.appendingPathComponent("comfy-z")
        let shared = root.appendingPathComponent("shared-qwen")
        for name in ["models/diffusion_models/z_image_turbo_bf16.safetensors", "models/vae/ae.safetensors"] {
            let path = comfy.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1, 2, 3]).write(to: path)
        }
        try check(ZImageInstallation.needsSharedText(comfy), "Comfy installation failed to request missing text components")
        try rejects { _ = try ZImageInstallation.install(model: comfy, sharedText: nil, directory: root.appendingPathComponent("bindings")) }
        for name in ["text_encoder/model.safetensors", "tokenizer/tokenizer.json"] {
            let path = shared.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1]).write(to: path)
        }
        try studio.installZImage(model: comfy, sharedText: shared)
        let installed = URL(fileURLWithPath: studio.draft.modelPath)
        try check(installed.appendingPathComponent("split_files").resolvingSymlinksInPath().path == comfy.appendingPathComponent("models").resolvingSymlinksInPath().path, "Comfy weights were not bound in place")
        try check(installed.appendingPathComponent("text_encoder").resolvingSymlinksInPath().path == shared.appendingPathComponent("text_encoder").resolvingSymlinksInPath().path, "Shared text weights were not bound in place")
        try check(!FileManager.default.fileExists(atPath: comfy.appendingPathComponent("tokenizer").path), "App modified the original model directory")
        let localProfileURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("assets/config/local-streaming-profile.json")
        let localProfile = try StudioLocalStreamingProfile.load(from: localProfileURL)
        let localHardware = #"{"gpu":"Apple M4 Pro","physical_memory_bytes":51539607552}"#
        let beforeLocalProfile = studio.draft
        studio.draft.operation = "image.generate"
        studio.draft.acceleration = nil
        studio.draft.profilePath = ""
        studio.draft.loras = []
        try check(localProfile.conflict(draft: studio.draft, systemJSON: localHardware) == nil,
                  "Local BF16 installation rejected experimental profile")
        try check(localProfile.conflict(draft: studio.draft, systemJSON: "{}") != nil,
                  "Local profile escaped its hardware scope")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
        try check(localProfile.conflict(draft: studio.draft, systemJSON: localHardware) != nil,
                  "Local profile accepted ANE")
        let rejectedDraft = try JSONEncoder().encode(studio.draft)
        studio.applyLocalStreamingProfile(localProfile, systemJSON: localHardware)
        let rejectedObject = try JSONSerialization.jsonObject(with: rejectedDraft) as! NSDictionary
        try check(rejectedObject == JSONSerialization.jsonObject(with: JSONEncoder().encode(studio.draft)) as! NSDictionary,
                  "Rejected local profile changed user settings")
        studio.draft.acceleration = nil
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        try check(localProfile.conflict(draft: studio.draft, systemJSON: localHardware) != nil,
                  "Local profile accepted LoRA")
        studio.draft.loras = []
        studio.draft.streaming.selection = .tier16
        let preservedPrompt = studio.draft.prompt
        let preservedSeed = studio.draft.seedText
        studio.applyLocalStreamingProfile(localProfile, systemJSON: localHardware)
        let localPair = try studio.draft.publicStreamingRequest(output: output)
        try check(localPair.v2 == nil && localPair.legacy.residency == "streamed" &&
                  localPair.legacy.memory_budget_bytes == 10 << 30 && studio.draft.width == 512 &&
                  studio.draft.height == 512 && studio.draft.steps == 9 && studio.draft.dynamicText,
                  "Local profile did not use the checked experimental request")
        try check(studio.draft.prompt == preservedPrompt && studio.draft.seedText == preservedSeed &&
                  studio.draft.modelPath == installed.path, "Local profile discarded user content")
        let savedLocalDraft = StudioState(directory: root).draft
        try check(savedLocalDraft.residency == "streamed" && savedLocalDraft.zImageStreamingBudgetGiB == 10 &&
                  savedLocalDraft.streaming.selection == .off, "Local profile did not persist independently of catalog tiers")
        var invalidProfile = try JSONSerialization.jsonObject(with: Data(contentsOf: localProfileURL)) as! [String: Any]
        invalidProfile["memoryGuarantee"] = true
        let invalidProfileURL = root.appendingPathComponent("invalid-local-profile.json")
        try JSONSerialization.data(withJSONObject: invalidProfile).write(to: invalidProfileURL)
        try rejects { _ = try StudioLocalStreamingProfile.load(from: invalidProfileURL) }
        studio.draft = beforeLocalProfile
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: loraManifest.path)
        let configuration = root.appendingPathComponent("app-configuration.json")
        studio.draft.modelPaths["flux2-klein-4b"] = ""
        try JSONEncoder().encode(studio.draft).write(to: configuration)
        studio.draft.modelPaths["flux2-klein-4b"] = shared.path
        studio.draft.loras = []; studio.draft.width = 1024
        try studio.importConfiguration(from: configuration)
        try check(studio.draft.loras.count == 1 && studio.draft.width == 512 && studio.draft.acceleration?.manifest == loraManifest.path,
                  "App configuration failed to restore LoRA, dimensions and acceleration")
        try check(studio.draft.modelPaths["flux2-klein-4b"] == shared.path,
                  "Empty configuration registration erased another installed model")
        let invalidConfiguration = root.appendingPathComponent("invalid-configuration.json")
        try Data("{}".utf8).write(to: invalidConfiguration)
        try rejects { try studio.importConfiguration(from: invalidConfiguration) }
        try check(studio.draft.loras.count == 1, "Rejected configuration changed the active draft")
        let previousSeed = studio.draft.seedText, previousRandomPolicy = studio.draft.randomSeed
        let previousModel = studio.draft.modelID, previousLoRAs = studio.draft.loras
        studio.newDraft()
        try check(studio.draft.seedText == previousSeed && studio.draft.randomSeed == previousRandomPolicy &&
                  studio.draft.modelID == previousModel && studio.draft.loras == previousLoRAs &&
                  studio.draft.assets.isEmpty && studio.draft.prompt.isEmpty, "New draft lost model preferences or retained its old content")
        let deletionRoot = root.appendingPathComponent("deletion")
        let deletionOutput = deletionRoot.appendingPathComponent("outputs/delete-test.png")
        try FileManager.default.createDirectory(at: deletionOutput.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: deletionOutput)
        let completedJob = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "deletion fixture", output: deletionOutput.path), state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1)
        try JSONEncoder().encode([completedJob]).write(to: deletionRoot.appendingPathComponent("jobs.json"))
        let deletionStore = NativeJobStore(directory: deletionRoot)
        try deletionStore.deleteJob(completedJob.id)
        try check(deletionStore.jobs.isEmpty && FileManager.default.fileExists(atPath: deletionOutput.path), "Deleting a task removed its image")
        try deletionStore.undoDeleteJob()
        try check(deletionStore.jobs.count == 1, "Undo failed to restore task")
        let trashed = try deletionStore.trashOutput(completedJob.id)
        try check(!FileManager.default.fileExists(atPath: deletionOutput.path) && !deletionStore.jobs[0].hasOutput, "Deleting an image left it visible")
        try check(NativeJobStore(directory: deletionRoot).jobs[0].outputDeleted == true, "Image deletion did not persist")
        if let trashed { try FileManager.default.moveItem(at: trashed, to: deletionOutput) }
        var externalJob = completedJob
        externalJob = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "external fixture", output: output.path), state: "succeeded", phase: "complete", completed: 1, total: 1, elapsed: 1)
        try JSONEncoder().encode([externalJob]).write(to: deletionRoot.appendingPathComponent("jobs.json"))
        let externalStore = NativeJobStore(directory: deletionRoot)
        try rejects { _ = try externalStore.trashOutput(externalJob.id) }
        try await StudioStreamingQueryTests.run(root: root.appendingPathComponent("worker-query-state"))
        print("PASS: seed policies, input roles/order/undo, clipboard, persistence, telemetry, FLUX9/H3/LTX/Wan/Z-Image defaults and separate LoRA forwarding")
    }

    /// Request/preflight coverage without model loading or AppKit image-provider services.
    @MainActor private static func verifyAccelerationRouting() async throws {
        func check(_ value: Bool, _ message: String) throws {
            guard value else { throw NativeFailure(message: message) }
        }
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-acceleration-routing-\(UUID())")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let adapter = root.appendingPathComponent("adapter.safetensors")
        try Data([1]).write(to: adapter)
        let output = root.appendingPathComponent("unused.png")
        for modelID in ["z-image-turbo"] {
            var draft = StudioDraft()
            draft.modelID = modelID; draft.modelPaths[modelID] = root.path
            draft.steps = modelID == "z-image-turbo" ? 8 : 4
            draft.loras = [StudioLoRA(path: adapter.path)]
            draft.loraStrategy = "inference_time"
            draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/missing/unused-manifest.json")
            let store = NativeJobStore(directory: root.appendingPathComponent(modelID))
            let resolved = try await store.resolveAcceleration(draft)
            let request = try resolved.request(output: output)
            try check(request.execution == "gpu" && request.ane_manifest == nil &&
                      draft.executionDeviceLabel == "GPU" && draft.aneConfigurationNotice?.contains("运行时加载") == true &&
                      store.accelerationStatus?.contains("GPU") == true && !store.resolvingAcceleration,
                      "\(modelID): runtime LoRA required an unused ANE partition or showed the wrong device")
            draft.loras[0].enabled = false
            try check(draft.executionDeviceLabel == "GPU + Core ML" && draft.aneConfigurationNotice == nil,
                      "\(modelID): disabled LoRA left a stale GPU notice")
        }
        let turbo = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
        try Data([1]).write(to: turbo)
        var qwen = StudioDraft()
        qwen.modelID = "qwen-image-2.1"; qwen.modelPaths[qwen.modelID] = root.path
        qwen.steps = 6; qwen.residency = "component_staged"
        qwen.loras = [StudioLoRA(path: turbo.path)]; qwen.loraStrategy = "inference_time"
        qwen.acceleration = StudioAcceleration(policy: "gpu_ane")
        let turboRequest = try qwen.request(output: output)
        try check(turboRequest.execution == "gpu" && turboRequest.ane_manifest == nil &&
                  qwen.executionDeviceLabel == "GPU" && qwen.aneConfigurationNotice?.contains("不会参与") == true,
                  "Qwen six-step LoRA device display disagreed with the GPU request")
        qwen.loras = [StudioLoRA(path: adapter.path)]; qwen.steps = 25
        try check(qwen.executionDeviceLabel == "GPU" && qwen.aneConfigurationNotice?.contains("请关闭 ANE") == true,
                  "Ordinary Qwen LoRA did not explain the invalid ANE combination")
        do {
            _ = try qwen.request(output: output)
            throw NativeFailure(message: "Ordinary Qwen LoRA unexpectedly bypassed the GPU admission gate")
        } catch {
            try check(error.localizedDescription.contains("仅支持纯 GPU"), "Unexpected ordinary Qwen LoRA admission error: \(error)")
        }
        qwen.acceleration?.policy = "gpu"
        try check((try qwen.request(output: output)).execution == "gpu" && qwen.aneConfigurationNotice == nil,
                  "Switching to GPU did not recover the ordinary Qwen LoRA request")
        qwen.loras = []; qwen.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/unused/manifest.json", qwen21W8A8: true)
        do {
            _ = try qwen.request(output: output, systemJSON: #"{"gpu":"Apple M4 Pro","physical_memory_bytes":51539607552}"#)
            throw NativeFailure(message: "Frozen Qwen W8A8 experiment was enabled on an unsupported device")
        } catch {
            try check(error.localizedDescription.contains("Apple M5 Pro"), "Frozen W8A8 hardware gate changed: \(error)")
        }
        let eligibleSystem = #"{"gpu":"Apple M4 Pro","physical_memory_bytes":51539607552}"#
        let runtimeStore = root.appendingPathComponent("runtime-tests")
        try fm.createDirectory(at: runtimeStore, withIntermediateDirectories: true)
        let zOptions = try RuntimeImageWorker.options(model: "z-image-turbo", store: runtimeStore)
        let qOptions = try RuntimeImageWorker.options(model: "qwen-image-2.1", store: runtimeStore)
        var z = StudioDraft()
        z.modelID = "z-image-turbo"; z.modelPaths[z.modelID] = root.path; z.steps = 8
        z.acceleration = StudioAcceleration(policy: "gpu_ane", runtimeAneProfileID: RuntimeImageWorker.profileID,
                                            runtimeDescriptor: zOptions.descriptor_path)
        for adapters in [[], [StudioLoRA(path: adapter.path)]] {
            z.loras = adapters; z.loraStrategy = "inference_time"
            let request = try z.request(output: output, systemJSON: eligibleSystem)
            try check(request.execution == "gpu_ane" && request.hybrid_mlp_mode == "runtime" &&
                      request.ane_manifest == zOptions.descriptor_path && !z.aneLoRARequiresGPU &&
                      z.executionDeviceLabel.contains("Private ANE"), "Explicit Z Runtime was forced onto frozen GPU-LoRA route")
        }
        z.steps = 9
        try check(z.runtimeANEIssue?.contains("8 步") == true, "Unverified Z Runtime steps admitted")
        qwen.loras = []; qwen.steps = 20; qwen.residency = "component_staged"
        qwen.acceleration = StudioAcceleration(policy: "gpu_ane", runtimeAneProfileID: RuntimeImageWorker.profileID,
                                               runtimeDescriptor: qOptions.descriptor_path)
        for count in 0...2 {
            qwen.assets = (0..<count).map { _ in StudioAsset(path: adapter.path, name: "CPU fixture", width: 32, height: 32) }
            qwen.operation = count == 0 ? "image.generate" : "image.edit"
            let request = try qwen.request(output: output, systemJSON: eligibleSystem)
            try check(request.execution == "gpu_ane" && request.hybrid_mlp_mode == "runtime" &&
                      request.qwen21_reference_size == 1024 && request.inputs?.count == count,
                      "Qwen Runtime base/reference request was normalized away")
        }
        qwen.loras = [StudioLoRA(path: turbo.path)]; qwen.steps = 6
        let runtimeTurbo = try qwen.request(output: output, systemJSON: eligibleSystem)
        try check(runtimeTurbo.execution == "gpu_ane" && runtimeTurbo.lora_strategy == "inference_time" &&
                  runtimeTurbo.ane_manifest == qOptions.descriptor_path && !qwen.aneLoRARequiresGPU,
                  "Qualified r128 Runtime request hit the old unconditional GPU guard")
        qwen.qwen21ReferenceSize = 512
        try check(qwen.runtimeANEIssue != nil, "Unverified Runtime reference encoding admitted")
        qwen.qwen21ReferenceSize = 1024; qwen.qwen21DiTCache = "balanced"
        try check(qwen.runtimeANEIssue != nil, "Unverified Runtime DiT cache combination admitted")
        qwen.qwen21DiTCache = "off"; qwen.assets = []; qwen.operation = "image.generate"
        qwen.loras = [StudioLoRA(path: adapter.path)]; qwen.steps = 20
        try check(qwen.runtimeANEIssue?.contains("r128") == true, "Ordinary Qwen adapter bypassed Runtime qualification")
        let decoded = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(qwen))
        try check(decoded.acceleration?.runtimeAneProfileID == RuntimeImageWorker.profileID, "Saved Runtime opt-in lost")
        let old = try JSONDecoder().decode(StudioAcceleration.self, from: Data(#"{"policy":"gpu","manifest":"","sourceManifest":""}"#.utf8))
        try check(old.runtimeAneProfileID == nil, "Old settings enabled Runtime by default")
        try await verifyRuntimeWorkerLifecycle(root: root.appendingPathComponent("runtime-lifecycle"))
        print("PASS acceleration routing: runtime LoRA GPU preflight, Qwen Turbo/ordinary LoRA recovery, disabled adapter state, frozen W8A8 hardware gate and explicit qualified Runtime routes; no model loaded")
    }

    @MainActor private static func verifyRuntimeWorkerLifecycle(root: URL) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        func check(_ value: Bool, _ reason: String) throws { if !value { throw NativeFailure(message: reason) } }
        let fixture = root.appendingPathComponent("fixture.png")
        let context = CGContext(data: nil, width: 512, height: 512, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 0.2, green: 0.3, blue: 0.4, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 512, height: 512))
        let writer = CGImageDestinationCreateWithURL(fixture as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(writer, context.makeImage()!, nil)
        try check(CGImageDestinationFinalize(writer), "Runtime CPU fixture encode failed")
        let png = try Data(contentsOf: fixture)
        func worker(_ mode: String) throws -> URL {
            let path = root.appendingPathComponent("runtime-fixture-\(mode).py")
            let code = """
            #!/usr/bin/python3
            import sys,json,base64,hashlib,time
            from pathlib import Path
            if sys.stdin.buffer.read(1)!=b'\\1':sys.exit(1)
            wire=json.loads(Path(sys.argv[2]).read_text());request=wire['native_request_v2'];out=request['outputs'][0]
            pixels=base64.b64decode('\(png.base64EncodedString())');Path(out['path']).write_bytes(pixels)
            if '\(mode)'=='hang':
                while True:time.sleep(0.1)
            metrics=dict(executor_backend=None,data_path='',partition_axis='rows',ane_channels=0,gpu_channels=0,
                         backend_fallback_reason='CPU fixture fallback',failure_reason='',hybrid_blocks_session_total=0,gpu_blocks_session_total=1,fallback_blocks_session_total=1)
            contract=dict(executor_backend=None,data_path='',partition_axis='rows')
            result=dict(schema_version=1,model=request['model'],operation=request['operation'],output=out['path'],width=512,height=512,
                        steps=request['sampling']['steps'],seed=request['sampling']['seed'],warmup=False,
                        plan=dict(execution='gpu',runtime_weight_contract=contract),hybrid=dict(runtime_weight=metrics,runtime_failed=False))
            receipt=dict(backend='gpu',selected_backend=None,actual_execution='gpu',partial_fallback=False,data_path='',partition_axis='',ane_channels=0,
                         fallback_reason='CPU fixture fallback',hybrid_blocks=0,gpu_blocks=1,fallback_blocks=1)
            terminal=dict(protocol_version=1,job_id=wire['job_id'],request_id=wire['request_id'],request_digest=wire['request_digest'],status='succeeded',
                          actual_container='cli_worker',runtime_fingerprint='\(NativeEngine.runtimeBuildIdentity())',resolution_digest=None,record_digest=None,
                          layout_digest=None,public_streaming_summary=None,error=None,runtime_options=wire['runtime_options'],runtime_receipt=receipt,result=result,
                          artifact=dict(path=out['path'],size=len(pixels),sha256=hashlib.sha256(pixels).hexdigest()))
            print(json.dumps(terminal))
            """
            try Data(code.utf8).write(to: path)
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: path.path)
            return path
        }
        func request(store: NativeJobStore, options: RuntimeImageWorker.Options) -> NativeRequest {
            var request = NativeRequest(prompt: "CPU fixture", output: store.directory.appendingPathComponent("outputs/final.png").path)
            request.model = "z-image-turbo"; request.operation = "image.generate"; request.steps = 8
            request.execution = "gpu_ane"; request.residency = "resident"; request.hybrid_mlp_mode = "runtime"
            request.allow_approximation = true; request.ane_manifest = options.descriptor_path
            return request
        }
        let success = NativeJobStore(directory: root.appendingPathComponent("success"), workerExecutable: try worker("success"))
        let options = try RuntimeImageWorker.options(model: "z-image-turbo", store: success.directory)
        let original = request(store: success, options: options)
        let job = try await success.generate(modelURL: root, request: original, runtimeOptions: options)
        try check(job.state == "succeeded" && job.runtimeWorker?.exitConfirmed == true && job.publicWorker == nil,
                  "Runtime worker did not persist its own confirmed exit")
        try check(job.routeSummary?.hasPrefix("GPU · Runtime 实验回退") == true && success.actualRoute == job.routeSummary,
                  "Runtime actual fallback display differs from its receipt")
        let published = try Data(contentsOf: URL(fileURLWithPath: original.output))
        try check(published == png, "Runtime transaction published different pixels")
        let reopened = NativeJobStore(directory: success.directory)
        try check(!reopened.busy && reopened.jobs[0].runtimeOptions == options && reopened.jobs[0].runtimeReceipt == job.runtimeReceipt,
                  "Runtime Options/Receipt did not restore after restart")
        guard let reference = reopened.jobs[0].runtimeWorker else { throw NativeFailure(message: "Runtime recovery lost worker reference") }
        let diagnostics = reference.directory(jobID: job.id, store: success.directory)
        let inputURL = diagnostics.appendingPathComponent("input.json")
        let input = try Data(contentsOf: inputURL)
        let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
        let intent = try JSONDecoder().decode(NativeRequestV2.self, from: JSONSerialization.data(withJSONObject: wire["native_request_v2"]!))
        let prepared = RuntimeImageWorker.Prepared(reference: reference, input: input, inputURL: inputURL, request: intent, options: reopened.jobs[0].runtimeOptions!)
        let terminal = try RuntimeImageWorker.validateTerminal(Data(contentsOf: diagnostics.appendingPathComponent("stdout.json")), prepared: prepared, exitCode: 0)
        try check(terminal.receipt == job.runtimeReceipt, "Restored Runtime binding could not reconstruct terminal validation")
        try await reopened.clearRuntimeProgramCache()
        try check(!fm.fileExists(atPath: options.cache_dir) && fm.fileExists(atPath: options.descriptor_path),
                  "Runtime cache cleanup removed descriptor or retained programs")

        let live = NativeJobStore(directory: root.appendingPathComponent("live"), workerExecutable: try worker("hang"))
        let liveOptions = try RuntimeImageWorker.options(model: "z-image-turbo", store: live.directory)
        let liveRequest = request(store: live, options: liveOptions)
        let task = Task { try await live.generate(modelURL: root, request: liveRequest, runtimeOptions: liveOptions) }
        var restored: NativeJobStore?
        do {
            var ready = false
            for _ in 0..<200 {
                if let reference = live.jobs.first?.runtimeWorker, fm.fileExists(atPath: reference.stagedOutput) { ready = true; break }
                try await Task.sleep(for: .milliseconds(20))
            }
            try check(ready, "Runtime CPU fake child never wrote staging")
            let recovered = NativeJobStore(directory: live.directory); restored = recovered
            try check(recovered.busy && recovered.workerCleanupPending && recovered.jobs.first?.state == "cleanup_pending",
                      "Restart admitted work while Runtime child was still alive")
            await recovered.refreshWorkerCleanup()
            try check(recovered.busy, "Live Runtime identity journal incorrectly released admission")
            do { try await live.clearRuntimeProgramCache(); throw NativeFailure(message: "Cache cleared while child alive") }
            catch let error as NativeFailure { try check(!error.message.hasPrefix("Cache cleared"), error.message) }
            // A second store has no persisted busy flag, but the shared runner
            // must still prevent manual cleanup while a child is active.
            let unaware = NativeJobStore(directory: root.appendingPathComponent("unaware"))
            let unawareOptions = try RuntimeImageWorker.options(model: "z-image-turbo", store: unaware.directory)
            do { try await unaware.clearRuntimeProgramCache(); throw NativeFailure(message: "Cache cleared despite shared active runner") }
            catch let error as NativeFailure { try check(!error.message.hasPrefix("Cache cleared"), error.message) }
            try check(!unaware.busy && fm.fileExists(atPath: unawareOptions.cache_dir),
                      "Rejected shared-runner cache cleanup changed contents or retained its busy latch")
        } catch {
            live.cancel(); _ = try? await task.value; throw error
        }
        live.cancel()
        do { _ = try await task.value; throw NativeFailure(message: "Cancelled Runtime CPU fixture succeeded") }
        catch is CancellationError {}
        try check(live.jobs[0].state == "cancelled" && live.jobs[0].runtimeWorker?.exitConfirmed == true && !live.busy,
                  "Runtime cancellation did not confirm exit and release admission")
        guard let restored else { throw NativeFailure(message: "Missing restored Runtime store") }
        await restored.refreshWorkerCleanup()
        try check(!restored.busy && !restored.workerCleanupPending && restored.jobs[0].state == "interrupted",
                  "Exited Runtime worker did not unblock recovered App")
        try check(!fm.fileExists(atPath: liveRequest.output), "Cancelled Runtime worker published output")
        let leftovers = try fm.contentsOfDirectory(atPath: live.directory.appendingPathComponent("outputs").path)
        try check(!leftovers.contains(where: { $0.hasPrefix(".tc-image-staging-") }), "Cancelled Runtime staging leaked")
        print("PASS Runtime CPU fake worker: publication, fallback display, Options/Reference/Receipt restart reconstruction, live-worker admission, cancellation/confirmed cleanup and owned cache cleanup")
    }

    @MainActor private static func verifyEditingWorkflows(root: URL, source: URL, png: Data) async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ reason: String) throws {
            if try !condition() { throw NativeFailure(message: reason) }
        }
        func snapshot(_ studio: StudioState) throws -> Data {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            return try encoder.encode(studio.draft)
        }
        func stagedCount(_ studio: StudioState) throws -> Int {
            let path = studio.importer.directory
            return FileManager.default.fileExists(atPath: path.path)
                ? try FileManager.default.contentsOfDirectory(atPath: path.path).count : 0
        }
        let studio = StudioState(directory: root)
        studio.selectModel("qwen-image-2.1")
        studio.draft.modelPaths[studio.draft.modelID] = root.path
        await studio.addFiles([source, source, source])
        let originals = studio.draft.assets
        try check(originals.count == 3, "Editing fixtures were not imported")
        studio.draft.initImageID = originals[1].id
        studio.useOnlyAssetForEditing(originals[2].id)
        try check(studio.draft.modelID == "qwen-image-2.1" && studio.draft.operation == "image.edit" &&
                  studio.draft.activeAssets == [originals[2]], "Single-image editing chose another Qwen input/model")
        studio.undoAssetChange()
        try check(studio.draft.assets == originals && studio.draft.initImageID == originals[1].id,
                  "Single-image undo did not restore reference order and primary selection")
        let adapter = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
        try Data([0]).write(to: adapter)
        try await verifyTurboControls(root: root.appendingPathComponent("turbo-controls"), source: source, adapter: adapter)
        studio.draft.loras = [StudioLoRA(path: adapter.path)]
        studio.applyQwen21TurboPreset()
        try check(studio.imageImportLimit == 3, "Turbo LoRA import limit differs from the native three-image contract")
        await studio.addFiles([source])
        try check(studio.draft.assets == originals, "Turbo file import admitted a fourth reference")
        let stroke = Qwen21AnnotationStroke(tool: .ellipse, points: [CGPoint(x: 0.1, y: 0.1), CGPoint(x: 0.9, y: 0.9)])
        let overflowMask = await studio.annotateQwen21Asset(originals[0].id, strokes: [stroke], output: .separateMask)
        try check(!overflowMask && studio.draft.assets == originals && studio.message?.contains("3") == true,
                  "Turbo mask bypassed the reference limit or reported the base limit")
        var request = NativeRequest(prompt: "Old generation prompt", output: source.path)
        request.model = "qwen-image-2.1"; request.operation = "image.generate"; request.steps = 6
        let job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "complete",
                            completed: 6, total: 6, elapsed: 1, modelPath: root.path)
        studio.draft.prompt = request.prompt
        let imported = await studio.editResult(job)
        try check(imported && studio.draft.prompt.isEmpty && studio.draft.operation == "image.edit" &&
                  studio.draft.assets.count == 1 && studio.draft.initImageID == studio.draft.assets[0].id &&
                  studio.draft.steps == 6 && studio.draft.loras[0].path == adapter.path,
                  "Continue editing kept the old prompt or lost the selected turbo setup")
        try check(studio.draft.assets[0].path != source.path &&
                  (try Data(contentsOf: URL(fileURLWithPath: studio.draft.assets[0].path))) == png &&
                  (try Data(contentsOf: source)) == png, "Continue editing changed the original result or its copy")
        studio.draft.prompt = "Change the teapot to cobalt blue"
        let editing = try studio.draft.request(output: root.appendingPathComponent("unused-edit.png"))
        try check(editing.inputs?.map(\.path) == studio.draft.assets.map(\.path) && editing.inputs?.first?.role == "reference" &&
                  editing.steps == 6, "Continue editing did not forward the copied result")
        studio.undoAssetChange()
        try check(studio.draft.assets == originals && studio.draft.prompt == editing.prompt,
                  "Result-input undo lost previous references or changed the new instruction")
        studio.draft.loras = []
        try check(studio.imageImportLimit == 10, "Disabling turbo did not restore base Qwen reference capacity")
        // A prompt entered while the result copy is pending is the next edit's
        // instruction. Context-preserving typing must neither erase it nor abort.
        studio.draft.prompt = "Previous request"
        let typing = studio.$importing.dropFirst().sink { importing in
            if importing { studio.draft.prompt = "New instruction entered during import" }
        }
        let typedResult = await studio.editResult(job)
        typing.cancel()
        try check(typedResult && studio.draft.prompt == "New instruction entered during import",
                  "Result import discarded or overwrote the new edit instruction")
        studio.draft.assets = originals; studio.draft.initImageID = originals[1].id
        let beforeConflict = try stagedCount(studio)
        let conflict = studio.$importing.dropFirst().sink { importing in
            if importing { studio.draft.assets.reverse() }
        }
        let conflicted = await studio.editResult(job)
        conflict.cancel()
        try check(!conflicted && studio.draft.assets == Array(originals.reversed()) &&
                  (try stagedCount(studio)) == beforeConflict, "Result import overwrote a changed tray or leaked its copy")
        studio.draft.assets = originals; studio.draft.initImageID = originals[1].id
        let annotationConflict = studio.$importing.dropFirst().sink { importing in
            if importing { studio.draft.assets.reverse() }
        }
        let annotated = await studio.annotateQwen21Asset(originals[0].id, strokes: [stroke])
        annotationConflict.cancel()
        try check(!annotated && studio.draft.assets == Array(originals.reversed()) &&
                  (try stagedCount(studio)) == beforeConflict, "Late annotation replaced another editing context")
        studio.draft.assets = originals; studio.draft.initImageID = originals[1].id
        let mask = await studio.annotateQwen21Asset(originals[1].id, strokes: [stroke], output: .separateMask)
        try check(mask && Array(studio.draft.assets.prefix(3)) == originals && studio.draft.initImageID == originals[1].id,
                  "Mask insertion renumbered existing references or changed the primary image")
        studio.undoAssetChange()
        try check(studio.message?.contains("已撤销") == true && studio.message?.contains("已添加") != true,
                  "Undoing a mask left its successful-addition message visible")
        // Deferred providers exercise an actual suspension, rather than relying
        // on file size or timing. Unrelated text/settings edits remain permitted.
        let deferred = DeferredStudioImageProvider()
        let pending = Task { await studio.importProviders([deferred.makeProvider()]) }
        try await deferred.waitUntilRequested()
        let lockedDraft = try snapshot(studio), lockedReset = studio.workspaceResetID
        studio.remove(originals[0].id); studio.move(originals[0].id, offset: 1)
        studio.undoAssetChange(); studio.selectModel("flux2-klein-4b"); studio.newDraft()
        try check(try snapshot(studio) == lockedDraft && studio.workspaceResetID == lockedReset,
                  "Asset/model/reset controls changed a pending import context")
        studio.draft.prompt = "Keep typing while the provider waits"; studio.draft.seedText = "123"
        deferred.finish(png); await pending.value
        try check(studio.draft.assets.count == 4 && Array(studio.draft.assets.prefix(3)) == originals &&
                  studio.draft.prompt == "Keep typing while the provider waits" && studio.draft.seedText == "123" && !studio.importing,
                  "Deferred import lost input order, text or unrelated sampling edits")
        for changedField in ["model", "operation", "assets", "primary", "cancel"] {
            studio.draft.modelID = "qwen-image-2.1"; studio.draft.operation = "image.edit"
            studio.draft.assets = originals; studio.draft.initImageID = originals[1].id
            let provider = DeferredStudioImageProvider(), before = try stagedCount(studio)
            let task = Task { await studio.importProviders([provider.makeProvider()]) }
            try await provider.waitUntilRequested()
            switch changedField {
            case "model": studio.draft.modelID = "flux2-klein-4b"
            case "operation": studio.draft.operation = "image.generate"
            case "assets": studio.draft.assets.reverse()
            case "primary": studio.draft.initImageID = originals[2].id
            default: task.cancel()
            }
            let changed = try snapshot(studio)
            provider.finish(png); await task.value
            try check(try snapshot(studio) == changed && (try stagedCount(studio)) == before && !studio.importing,
                      "Deferred \(changedField) change was overwritten or left a staged file/locked controls")
        }
        // Reserve the UI task synchronously, before its async import body can
        // run. Draft/input mutation must already be locked in this window.
        studio.draft.modelID = "qwen-image-2.1"; studio.draft.operation = "image.edit"
        studio.draft.assets = originals; studio.draft.initImageID = originals[1].id
        let reservedBefore = try snapshot(studio), resetBefore = studio.workspaceResetID
        let notStarted = DeferredStudioImageProvider()
        try check(studio.beginImageImportProviders([notStarted.makeProvider()]), "Cannot reserve a UI provider import")
        guard let notStartedTask = studio.imageImportTask else { throw NativeFailure(message: "Reserved import lost its task handle") }
        try check(studio.imageInputsBusy && !studio.importing, "UI task did not lock inputs before beginning its async work")
        studio.remove(originals[0].id); studio.move(originals[0].id, offset: 1)
        studio.undoAssetChange(); studio.selectModel("flux2-klein-4b"); studio.newDraft()
        let blockedPreparation = await studio.prepareAsset(id: originals[0].id, preset: .fit512)
        try check(!blockedPreparation && (try snapshot(studio)) == reservedBefore && studio.workspaceResetID == resetBefore,
                  "A reserved import allowed deleting/resetting/resizing its reference context")
        studio.cancelImageImport(); await notStartedTask.value
        try check(!studio.imageInputsBusy && studio.imageImportTask == nil && !studio.cancellingImageImport,
                  "Cancellation before the import body began did not release its task handle")

        // Unlike the earlier cancel case, the provider deliberately does NOT
        // finish. Cancellation must complete without a callback or staged copy.
        let stalled = DeferredStudioImageProvider(), filesBeforeCancel = try stagedCount(studio)
        let draftBeforeCancel = try snapshot(studio)
        var cancellationFinished = false
        let cancelledImport = Task {
            await studio.importProviders([stalled.makeProvider()])
            cancellationFinished = true
        }
        try await stalled.waitUntilRequested()
        cancelledImport.cancel()
        let cancellationStart = ContinuousClock.now
        while !cancellationFinished && cancellationStart.duration(to: .now) < .seconds(2) {
            try await Task.sleep(for: .milliseconds(5))
        }
        try check(cancellationFinished && !studio.imageInputsBusy && (try snapshot(studio)) == draftBeforeCancel &&
                  (try stagedCount(studio)) == filesBeforeCancel && studio.message?.contains("已取消") == true,
                  "A provider without a callback kept the cancelled import locked or changed its inputs")
        await cancelledImport.value

        // An old provider can still attempt its callback after a fresh retry
        // starts. It must neither append its old image nor unlock the new task.
        let retryProvider = DeferredStudioImageProvider()
        try check(studio.beginImageImportProviders([retryProvider.makeProvider()]), "Cannot retry after cancelling a stalled provider")
        guard let retryTask = studio.imageImportTask else { throw NativeFailure(message: "Retry lost its task handle") }
        try await retryProvider.waitUntilRequested()
        try check(!studio.beginImageImportProviders([notStarted.makeProvider()]), "A second UI import replaced a pending retry")
        stalled.finish(png)
        try await Task.sleep(for: .milliseconds(50))
        try check(studio.importing && studio.imageImportTask != nil && (try snapshot(studio)) == draftBeforeCancel &&
                  (try stagedCount(studio)) == filesBeforeCancel,
                  "A late cancelled-provider callback changed references or cleared the retry lock")
        retryProvider.finish(png); await retryTask.value
        try check(!studio.imageInputsBusy && studio.imageImportTask == nil && studio.draft.assets.count == originals.count + 1 &&
                  Array(studio.draft.assets.prefix(originals.count)) == originals,
                  "A successful retry lost original reference order or failed to release its task handle")
        studio.selectModel("flux2-klein-4b")
        studio.draft.assets = originals; studio.draft.initImageID = originals[0].id
        studio.draft.prompt = "Modify the second image"
        studio.useOnlyAssetForEditing(originals[1].id)
        try check(studio.draft.operation == "image.transform" && studio.draft.activeAssets == [originals[1]] &&
                  studio.draft.assets == originals, "FLUX single-image editing selected a wrong input or removed unused references")
        studio.selectModel("qwen-image-2.1")
        studio.draft.loras = [StudioLoRA(path: adapter.path)]; studio.applyQwen21TurboPreset()
        studio.draft.operation = "image.edit"; studio.draft.assets = originals; studio.draft.initImageID = originals[2].id
        studio.draft.seedText = "765"; studio.draft.randomSeed = true; studio.draft.dynamicText = true
        studio.draft.strength = 0.42; studio.lastSeed = 987
        let keptLoRAs = studio.draft.loras
        var reset = studio.workspaceResetID
        studio.message = "Previous failure"; studio.newDraft()
        try check(studio.workspaceResetID != reset && studio.message == nil && studio.draft.assets.isEmpty && !studio.canUndoAssets,
                  "New draft did not clear edit state or emit a workspace reset")
        try check(studio.draft.modelID == "qwen-image-2.1" && studio.draft.modelPath == root.path &&
                  studio.draft.operation == "image.generate" && studio.draft.prompt.isEmpty && studio.draft.initImageID == nil &&
                  studio.draft.loras == keptLoRAs && studio.draft.steps == 6 && studio.draft.width == 512 && studio.draft.height == 512 &&
                  studio.draft.acceleration?.policy == "gpu" && studio.draft.residency == "component_staged" &&
                  studio.draft.seedText == "765" && studio.draft.randomSeed && studio.draft.dynamicText &&
                  studio.draft.strength == 0.42 && studio.lastSeed == nil,
                  "New draft switched away from Qwen turbo or reset generation preferences")
        reset = studio.workspaceResetID
        studio.newDraft()
        try check(studio.workspaceResetID != reset, "Resetting an already empty draft did not notify the workspace")
        studio.selectModel("ltx-2.5-distilled")
        studio.draft.operation = "video.image"; studio.draft.assets = originals
        studio.draft.width = 832; studio.draft.height = 480; studio.draft.frames = 97; studio.draft.steps = 11
        studio.draft.ltxBackend = "c_metal"; studio.draft.ltxFastAV = false
        studio.newDraft()
        try check(studio.draft.modelID == "ltx-2.5-distilled" && studio.draft.operation == "video.generate" &&
                  studio.draft.assets.isEmpty && studio.draft.width == 832 && studio.draft.height == 480 &&
                  studio.draft.frames == 97 && studio.draft.steps == 11 && studio.draft.ltxBackend == "c_metal" && !studio.draft.ltxFastAV,
                  "New video draft lost its model or sampling/backend preferences")
        reset = studio.workspaceResetID; studio.reuse(job)
        try check(studio.workspaceResetID != reset && studio.draft.prompt == job.request.prompt,
                  "History reuse did not reset the workspace or incorrectly cleared its stored prompt")
        let configuration = root.appendingPathComponent("editing-config.json")
        try JSONEncoder().encode(studio.draft).write(to: configuration)
        reset = studio.workspaceResetID
        try studio.importConfiguration(from: configuration)
        try check(studio.workspaceResetID != reset, "Configuration import did not reset the workspace")
        print("PASS editing state: result copies/fresh instructions, single-image selection, reference/mask order, turbo limits, delayed-provider conflicts/cancellation, prompt preservation and workspace resets")
    }

    @MainActor private static func verifyTurboControls(root: URL, source: URL, adapter: URL) async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ reason: String) throws {
            if try !condition() { throw NativeFailure(message: reason) }
        }
        let studio = StudioState(directory: root)
        studio.selectModel("qwen-image-2.1")
        studio.draft.modelPaths[studio.draft.modelID] = root.path
        await studio.addFiles(Array(repeating: source, count: 4))
        let assets = studio.draft.assets
        studio.draft.prompt = "Preserve this description and the retained reference tray"
        let prompt = studio.draft.prompt
        studio.draft.width = 768; studio.draft.steps = 40
        studio.addLoRA(adapter.path)
        let id = studio.draft.loras[0].id
        try check(studio.draft.qwen21TurboConfigurationIssues.isEmpty && studio.draft.qwen21TurboSummary == "Turbo · 6 步" &&
                  studio.draft.steps == 6 && studio.draft.width == 512 && studio.draft.height == 512 &&
                  studio.draft.prompt == prompt && studio.draft.assets == assets,
                  "Adding the supported Turbo adapter did not apply its setup or lost draft content")
        let text = try studio.draft.request(output: root.appendingPathComponent("unused-text.png"))
        try check(text.inputs?.isEmpty == true && studio.draft.assets.count == 4 && text.execution == "gpu" && text.steps == 6,
                  "Text generation incorrectly used or rejected retained reference images")
        studio.addLoRA(adapter.deletingLastPathComponent().appendingPathComponent("./" + adapter.lastPathComponent).path)
        try check(studio.draft.loras.count == 1 && studio.draft.loras[0].id == id,
                  "Adding an equivalent adapter path created a duplicate")
        let second = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors")
        try Data([0]).write(to: second)
        studio.addLoRA(second.path)
        let secondID = studio.draft.loras.last!.id
        try check(studio.draft.loras.count == 2 && studio.draft.qwen21TurboSummary == nil && studio.draft.steps == 40,
                  "Multiple active adapters were advertised as a supported Turbo configuration")
        do {
            try studio.draft.validate()
            throw NativeFailure(message: "Multiple Turbo adapters unexpectedly validated")
        } catch {
            let reason = error.localizedDescription
            let explainsIsolation = ["混用", "单个", "一个", "单独"].contains { reason.contains($0) }
            try check(reason.contains("适配器") && explainsIsolation,
                      "Multiple adapters lost their actionable validation reason: \(reason)")
        }
        studio.removeLoRA(secondID)
        try check(studio.draft.loras.count == 1 && studio.draft.loras[0].id == id && studio.draft.steps == 6,
                  "Removing an extra adapter did not restore the single-adapter Turbo configuration")
        studio.changeModel("flux2-klein-4b"); studio.changeModel("qwen-image-2.1")
        try check(studio.draft.loras.first?.id == id && studio.draft.steps == 6 &&
                  studio.draft.qwen21TurboConfigurationIssues.isEmpty && studio.draft.prompt == prompt && studio.draft.assets == assets,
                  "Model round trip restored Turbo LoRA with the base 40-step schedule")
        studio.setLoRAEnabled(id, enabled: false)
        try check(studio.draft.steps == 40 && studio.draft.qwen21TurboSummary == nil &&
                  studio.draft.loras.count == 1 && !studio.draft.loras[0].enabled &&
                  studio.draft.prompt == prompt && studio.draft.assets == assets,
                  "Disabling Turbo lost its file/content or left the base model on six steps")
        let base = try studio.draft.request(output: root.appendingPathComponent("unused-base.png"))
        try check(base.steps == 40 && base.loras == nil && base.inputs?.isEmpty == true,
                  "A disabled adapter or retained references leaked into base text generation")
        let qwenDescriptor = studio.models.first { $0.id == "qwen-image-2.1" }!
        var pe = studio.draft
        pe.promptEnhance = true; pe.promptEnhancerPath = root.appendingPathComponent("missing-pe").path
        do {
            try pe.validate(model: qwenDescriptor)
            throw NativeFailure(message: "Missing PE installation unexpectedly validated")
        } catch {
            try check(error.localizedDescription.contains("PE-T2I"), "Missing PE directory was not exposed by settings validation")
        }
        let peDirectory = root.appendingPathComponent("pe-fixture")
        try FileManager.default.createDirectory(at: peDirectory, withIntermediateDirectories: true)
        try "Rewrite the supplied description".write(to: peDirectory.appendingPathComponent("system_prompt.txt"), atomically: true, encoding: .utf8)
        try "{}".write(to: peDirectory.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        pe.promptEnhancerPath = peDirectory.path
        try pe.validate(model: qwenDescriptor)
        pe.operation = "image.edit"; pe.assets = Array(assets.prefix(1)); pe.promptEnhanceEditExperimental = false
        do {
            try pe.validate(model: qwenDescriptor)
            throw NativeFailure(message: "PE editing without experimental opt-in unexpectedly validated")
        } catch {
            try check(error.localizedDescription.contains("显式开启"), "PE editing opt-in was not exposed by settings validation")
        }
        pe.promptEnhanceEditExperimental = true
        try pe.validate(model: qwenDescriptor)
        let peRequest = try pe.request(output: root.appendingPathComponent("unused-pe-edit.png"))
        try check(peRequest.prompt_enhance == true && peRequest.prompt_enhance_edit_experimental == true &&
                  peRequest.prompt_enhancer_path == peDirectory.path,
                  "Valid PE settings did not preserve the request's enhancement/experimental intent")
        studio.setLoRAEnabled(id, enabled: true)
        try check(studio.draft.steps == 6 && studio.draft.qwen21TurboConfigurationIssues.isEmpty,
                  "Re-enabling Turbo did not restore its qualified schedule")
        studio.draft.steps = 40
        try check(studio.draft.qwen21TurboSummary == "Turbo · 待配置" &&
                  studio.draft.qwen21TurboConfigurationIssues.count == 1 &&
                  studio.draft.qwen21TurboConfigurationIssues[0].contains("40") &&
                  !studio.draft.qwen21TurboConfigurationIssues[0].contains("512"),
                  "A steps-only conflict falsely blamed the already correct canvas")
        do {
            try studio.draft.validate()
            throw NativeFailure(message: "Turbo base schedule unexpectedly validated")
        } catch {
            try check(error.localizedDescription.contains("40") && !error.localizedDescription.contains("512"),
                      "Request validation diverged from the precise steps-only reason")
        }
        studio.save()
        let reopened = StudioState(directory: root)
        try check(reopened.draft.steps == 40 && reopened.draft.qwen21TurboSummary == "Turbo · 待配置",
                  "Opening an old Turbo draft silently migrated its settings")
        studio.draft.steps = 6; studio.draft.width = 768
        studio.draft.loras[0].role = "text_encoder"; studio.draft.loras[0].strength = 0.8
        studio.draft.loraStrategy = "disk_premerge"
        let issues = studio.draft.qwen21TurboConfigurationIssues
        try check(issues.count == 4 && issues.contains(where: { $0.contains("768×512") && $0.contains("App") }) &&
                  issues.contains(where: { $0.contains("text_encoder") }) && issues.contains(where: { $0.contains("0.8") }) &&
                  issues.contains(where: { $0.contains("disk_premerge") }), "Turbo issues omitted actual canvas/role/strength/strategy values")
        studio.applyQwen21TurboPreset()
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
        try check(studio.draft.accelerationHint == "GPU · BF16 · LoRA 运行时加载" &&
                  studio.draft.executionDeviceLabel == "GPU" &&
                  studio.draft.aneConfigurationNotice?.contains("不会参与") == true &&
                  (try studio.draft.request(output: root.appendingPathComponent("unused-gpu.png"))).execution == "gpu",
                  "Turbo device hint disagreed with the forced pure GPU request")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu")
        studio.draft.loraStrategy = "auto"
        try check(studio.draft.qwen21TurboConfigurationIssues.isEmpty,
                  "The supported automatic strategy was treated as a conflict")
        studio.removeLoRA(id)
        try check(studio.draft.steps == 40 && studio.draft.loras.isEmpty && studio.draft.qwen21TurboSummary == nil &&
                  studio.draft.prompt == prompt && studio.draft.assets == assets,
                  "Removing the last Turbo adapter left six-step base settings or erased content")
        studio.addLoRA(adapter.path)
        studio.draft.steps = 12
        studio.setLoRAEnabled(studio.draft.loras[0].id, enabled: false)
        try check(studio.draft.steps == 12, "Leaving Turbo reset a manually chosen non-six-step schedule")
        studio.changeModel("flux2-klein-4b")
        studio.draft.steps = 13; studio.draft.width = 768
        studio.addLoRA(adapter.path)
        let fluxID = studio.draft.loras[0].id
        studio.setLoRAEnabled(fluxID, enabled: false); studio.setLoRAEnabled(fluxID, enabled: true); studio.removeLoRA(fluxID)
        try check(studio.draft.steps == 13 && studio.draft.width == 768 && studio.draft.qwen21TurboSummary == nil &&
                  studio.draft.prompt == prompt && studio.draft.assets == assets,
                  "Qwen Turbo controls changed another model's schedule or content")
        print("PASS Turbo controls: add/enable/disable/remove, model round trip, retained TTI references, precise conflict reasons, GPU hint and unchanged old drafts/other models")
    }
}
