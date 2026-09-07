import AppKit
import Foundation

@main
struct StudioBehaviorTests {
    @MainActor static func main() async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try condition() else { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected validation failure")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-studio-test-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let studio = StudioState(directory: root)
        studio.draft.modelPaths["flux2-klein-4b"] = "/test/model"
        let output = root.appendingPathComponent("output.png")
        try check(try studio.draft.request(output: output).seed == 42, "Default seed changed")
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
        manifest["source"] = ["checkpoint": weight.path, "checkpoint_bytes": 4]
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path, cache: compiled) == nil, "Wrong checkpoint accepted")
        manifest["source"] = ["checkpoint": weight.path, "checkpoint_bytes": 3]
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestFile)
        try FileManager.default.removeItem(at: compiled.appendingPathComponent("block19.mlmodelc"))
        try check(AccelerationDiscovery.find(modelPath: fixture.path, preferred: manifestFile.path, cache: compiled) == nil, "Incomplete partition accepted")
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
        func event(_ sequence: Int, _ completed: Int, _ seconds: Double, _ phase: String = "denoise") -> NativeEvent { NativeEvent(sequence: sequence, phase: phase, completed: completed, total: 4, elapsed_seconds: seconds) }
        try check(telemetry.observe(event(1, 1, 10)) == nil, "Transform offset counted as completed sample")
        try check(telemetry.observe(event(2, 2, 12)) == nil, "Single sample shown as reliable speed")
        _ = telemetry.observe(event(3, 2, 13)); _ = telemetry.observe(event(4, 30, 13.5, "block"))
        try check(telemetry.observe(event(5, 3, 15)) == 2.5, "Step duration includes wrong boundaries")
        try check(telemetry.observe(event(2, 4, 50)) == 2.5, "Out-of-order telemetry altered speed")
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
        studio.changeOperation("video.image"); studio.draft.initImageID = studio.draft.assets.first?.id
        let ltx = try studio.draft.request(output: root.appendingPathComponent("ltx.mp4"))
        try check(ltx.width == 704 && ltx.height == 448 && ltx.frames == 97 && ltx.steps == 11 && ltx.inputs?.first?.role == "first_frame",
                  "LTX descriptor defaults or first-frame mapping changed")
        studio.selectModel("fastmetal-1.3b-qad")
        let lora = root.appendingPathComponent("adapter.safetensors"); try Data([9]).write(to: lora)
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        let fastmetal = try studio.draft.request(output: root.appendingPathComponent("fastmetal.mp4"))
        try check(fastmetal.width == 832 && fastmetal.height == 480 && fastmetal.frames == 81 && fastmetal.fps == 16 && fastmetal.execution == "gpu" && fastmetal.loras?.count == 1,
                  "FastMetal defaults or separate LoRA forwarding changed")
        studio.selectModel("flux2-klein-4b")
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: "/test/compiled/manifest.json")
        let fluxLoRA = try studio.draft.request(output: root.appendingPathComponent("flux-lora.png"))
        try check(fluxLoRA.execution == "gpu" && fluxLoRA.ane_manifest == nil && fluxLoRA.loras?.count == 1,
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
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
        let zImage = try studio.draft.request(output: root.appendingPathComponent("z-image.png"))
        try check(zImage.model == "z-image-turbo" && zImage.operation == "image.generate" &&
                    zImage.width == 1024 && zImage.height == 1024 && zImage.steps == 9 &&
                    zImage.frames == 1 && zImage.audio == false && zImage.execution == "gpu" &&
                    zImage.loras?.first?.role == "transformer",
                  "Z-Image App defaults, GPU fail-closed policy or separate LoRA forwarding changed")
        studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane", manifest: loraManifest.path)
        studio.draft.loras = [StudioLoRA(path: lora.path, strength: 0.8)]
        let zImageLoRAHybrid = try studio.draft.request(output: root.appendingPathComponent("z-image-lora-hybrid.png"))
        try check(zImageLoRAHybrid.execution == "gpu_ane" &&
                    zImageLoRAHybrid.ane_manifest == loraManifest.path,
                  "Z-Image LoRA-bound ANE artifact was not forwarded")
        studio.newDraft()
        try check(studio.draft.seedText == "42" && !studio.draft.randomSeed && studio.draft.assets.isEmpty, "New draft defaults failed")
        print("PASS: seed policies, input roles/order/undo, clipboard, persistence, telemetry, FLUX9/H3/LTX/FastMetal/Z-Image defaults and separate LoRA forwarding")
    }
}
