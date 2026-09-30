import AppKit
import Foundation
import UniformTypeIdentifiers
import ImageIO

// CPU-only App contract tests. No model weights or generation-quality claim.
@main struct Qwen21WorkflowTests {
    @MainActor static func main() async throws {
        func check(_ value: @autoclosure () throws -> Bool, _ reason: String) throws {
            if try !value() { throw NativeFailure(message: reason) }
        }
        // Export reproducible fixtures through the SAME renderer as the App.
        // This mode performs no inference and never overwrites existing files.
        if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--export-guidance-fixtures" {
            let source = URL(fileURLWithPath: CommandLine.arguments[2])
            let directory = URL(fileURLWithPath: CommandLine.arguments[3])
            let targets = [directory.appendingPathComponent("lid-annotation.png"),
                           directory.appendingPathComponent("lid-mask.png")]
            try check(targets.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) }, "Guidance fixtures already exist")
            let lid = Qwen21AnnotationStroke(tool: .ellipse,
                points: [CGPoint(x: 0.16, y: 0.055), CGPoint(x: 0.69, y: 0.425)], width: 0.012)
            let annotated = try Qwen21AnnotationRenderer.render(source: source, strokes: [lid])
            let mask = try Qwen21AnnotationRenderer.render(source: source, strokes: [lid], output: .separateMask)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try annotated.write(to: targets[0], options: .withoutOverwriting)
            try mask.write(to: targets[1], options: .withoutOverwriting)
            print("Exported App-rendered lid annotation and separate mask to \(directory.path)")
            return
        }
        try check(CommandLine.arguments.count == 1, "Usage: [--export-guidance-fixtures source output-directory]")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-qwen21-workflows-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 32,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        memset(bitmap.bitmapData!, 255, bitmap.bytesPerRow * bitmap.pixelsHigh)
        bitmap.setColor(NSColor(deviceRed: 0, green: 0, blue: 0, alpha: 0), atX: 0, y: 0)
        bitmap.setColor(NSColor(deviceRed: 0, green: 0, blue: 0, alpha: 1), atX: 8, y: 8)
        let data = bitmap.representation(using: .png, properties: [:])!
        let fixture = root.appendingPathComponent("fixture.png")
        try data.write(to: fixture)
        let studio = StudioState(directory: root.appendingPathComponent("state"))
        studio.selectModel("qwen-image-2.1")
        try check(studio.draft.width == 512 && studio.draft.height == 512,
                  "Qwen21 default left the maintained 512-square scope")
        studio.draft.modelPaths["qwen-image-2.1"] = root.path
        studio.draft.prompt = "Arrange these ten reference subjects in order."
        studio.changeOperation("image.edit")
        try check(studio.imageImportLimit == 10, "Qwen21 App still has an eight-image import cap")
        await studio.addFiles(Array(repeating: fixture, count: 10))
        try check(studio.draft.assets.count == 10, "Ten-image file import failed")
        let original = studio.draft.assets
        let output = root.appendingPathComponent("unused.png")
        try await verifyTurboAndEditing(root: root, fixture: fixture, assets: original)
        try verifyDiTCacheAndOrdinaryLoRA(root: root, fixture: fixture, assets: original)
        let expectedCanvases = [(2048, 2048), (2400, 1792), (1792, 2400),
                                (2528, 1696), (1696, 2528), (2752, 1536), (1536, 2752)]
        try check(Qwen21CanvasPreset.recommended.count == expectedCanvases.count,
                  "Official Qwen21 canvas presets missing")
        for (preset, expected) in zip(Qwen21CanvasPreset.recommended, expectedCanvases) {
            try check(preset.width == expected.0 && preset.height == expected.1,
                      "Qwen21 preset differs from official aligned dimensions")
            var draft = studio.draft
            draft.operation = "image.generate"; draft.acceleration = StudioAcceleration(policy: "gpu")
            draft.width = preset.width; draft.height = preset.height
            let request = try draft.request(output: output)
            try check(request.width == preset.width && request.height == preset.height,
                      "App altered Qwen21 canvas preset")
            _ = try NativeEngine.plan(request)
        }
        var peDraft = studio.draft
        peDraft.operation = "image.generate"
        peDraft.promptEnhance = true
        peDraft.promptEnhancerPath = root.appendingPathComponent("pe-fixture").path
        let peDirectory = URL(fileURLWithPath: peDraft.promptEnhancerPath)
        try FileManager.default.createDirectory(at: peDirectory, withIntermediateDirectories: true)
        try Data("system".utf8).write(to: peDirectory.appendingPathComponent("system_prompt.txt"))
        try Data("{}".utf8).write(to: peDirectory.appendingPathComponent("tokenizer.json"))
        let peRequest = try peDraft.request(output: output)
        try check(peRequest.prompt_enhance_edit_experimental == false,
                  "Text-only PE accidentally enabled experimental image route")
        try check(peRequest.prompt_enhance == true && peRequest.prompt_enhancer_path == peDraft.promptEnhancerPath,
                  "PE request lost installation/enable fields")
        _ = try NativeEngine.plan(peRequest)
        let savedPEDraft = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(peDraft))
        try check(savedPEDraft.promptEnhance && savedPEDraft.promptEnhancerPath == peDraft.promptEnhancerPath,
                  "PE draft fields were not persisted")
        peDraft.operation = "image.edit"
        var editPERejected = false
        do { _ = try peDraft.request(output: output) } catch { editPERejected = true }
        try check(editPERejected, "Visual PE accepted without explicit experimental opt-in")
        peDraft.promptEnhanceEditExperimental = true
        let editPERequest = try peDraft.request(output: output)
        try check(editPERequest.prompt_enhance == true && editPERequest.prompt_enhance_edit_experimental == true &&
                  editPERequest.inputs?.map(\.path) == original.map(\.path),
                  "Experimental PE-I2I lost flags or ten-reference order")
        _ = try NativeEngine.plan(editPERequest)
        let savedEditPE = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(peDraft))
        try check(savedEditPE.promptEnhanceEditExperimental, "Experimental edit PE opt-in was not persisted")
        let legacyPEDraft = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(!legacyPEDraft.promptEnhanceEditExperimental, "Legacy draft defaulted to experimental edit PE")
        peDraft.operation = "image.generate"
        try check(try peDraft.request(output: output).prompt_enhance_edit_experimental == false,
                  "Switching to T2I retained edit-only flag")
        peDraft.operation = "image.edit"; peDraft.promptEnhance = false
        try check(try peDraft.request(output: output).prompt_enhance_edit_experimental == false,
                  "Disabled PE retained experimental edit flag")
        let request = try studio.draft.request(output: output)
        try check(request.inputs?.map(\.path) == original.map(\.path), "Ten-reference order was not preserved")
        _ = try NativeEngine.plan(request)
        await studio.addFiles([fixture])
        try check(studio.draft.assets == original, "Overflow file import changed the draft")
        studio.draft.assets.append(original[0])
        var rejected = false
        do { _ = try studio.draft.request(output: output) } catch { rejected = true }
        try check(rejected, "Eleven-reference request was accepted")
        studio.draft.assets = original
        studio.move(original[9].id, offset: -1)
        try check(studio.draft.assets[8].id == original[9].id, "Tenth reference cannot be reordered")
        studio.undoAssetChange()
        try check(studio.draft.assets == original, "Reference reorder undo failed")
        studio.save()
        let restored = StudioState(directory: root.appendingPathComponent("state"))
        try check(restored.draft.assets == original, "Ten references were not persisted")

        studio.draft.assets = Array(original.prefix(9))
        let board = NSPasteboard.withUniqueName()
        defer { board.releaseGlobally() }
        board.setData(data, forType: .png)
        await studio.pasteImage(from: board)
        try check(studio.draft.assets.count == 10, "Pasting the tenth reference failed")
        let pasted = studio.draft.assets
        await studio.pasteImage(from: board)
        try check(studio.draft.assets == pasted, "Overflow paste changed the draft")

        studio.draft.assets = []
        let providers = (0..<10).map { _ in NSItemProvider(item: data as NSData, typeIdentifier: UTType.png.identifier) }
        await studio.importProviders(providers)
        try check(studio.draft.assets.count == 10, "Ten-image provider import failed")
        let dropped = studio.draft.assets
        await studio.importProviders([providers[0]])
        try check(studio.draft.assets == dropped, "Overflow provider import changed the draft")

        studio.draft.assets = []
        let before = studio.draft.prompt
        studio.applyQwen21Example(.maskEdit)
        try check(studio.draft.prompt == before, "Missing mask references changed the prompt")
        studio.draft.assets = Array(original.prefix(2))
        let dot = Qwen21AnnotationStroke(tool: .brush, points: [CGPoint(x: 0.75, y: 0.25)], width: 0.1)
        let annotatedData = try Qwen21AnnotationRenderer.render(source: fixture, strokes: [dot])
        let annotated = NSBitmapImageRep(data: annotatedData)!
        func rgba(_ image: NSBitmapImageRep, _ x: Int, _ y: Int) -> NSColor {
            image.colorAt(x: x, y: y)!.usingColorSpace(.sRGB)!
        }
        // A 3.2-pixel round brush is antialiased; test red dominance/location,
        // not exact byte equality at a subpixel curved boundary.
        try check(rgba(annotated, 48, 8).redComponent > 0.95 && rgba(annotated, 48, 8).greenComponent < 0.2,
                  "Annotation geometry \(annotated.pixelsWide)x\(annotated.pixelsHigh): top \(rgba(annotated, 48, 8)), bottom \(rgba(annotated, 48, 24))")
        try check(rgba(annotated, 48, 24).greenComponent > 0.95, "Annotation appeared in the wrong vertical location")
        try check(rgba(annotated, 8, 8).redComponent < 0.05 && rgba(annotated, 0, 0).alphaComponent < 0.05,
                  "Annotation renderer changed source orientation or transparency")
        let ellipse = Qwen21AnnotationStroke(tool: .ellipse,
            points: [CGPoint(x: 0.25, y: 0.25), CGPoint(x: 0.75, y: 0.75)], width: 0.1)
        let circled = NSBitmapImageRep(data: try Qwen21AnnotationRenderer.render(source: fixture, strokes: [ellipse]))!
        try check(rgba(circled, 32, 8).greenComponent < 0.2 && rgba(circled, 32, 16).greenComponent > 0.95,
                  "Ellipse is missing or filled instead of outlined")
        let rotated = root.appendingPathComponent("rotated.tiff")
        let destination = CGImageDestinationCreateWithURL(rotated as CFURL, UTType.tiff.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, bitmap.cgImage!, [kCGImagePropertyOrientation: 6] as CFDictionary)
        try check(CGImageDestinationFinalize(destination), "Cannot save EXIF orientation fixture")
        let oriented = NSBitmapImageRep(data: try Qwen21AnnotationRenderer.render(source: rotated, strokes: [dot]))!
        try check(oriented.pixelsWide == 32 && oriented.pixelsHigh == 64, "Annotation lost EXIF orientation")
        let maskData = try Qwen21AnnotationRenderer.render(source: fixture, strokes: [ellipse], output: .separateMask)
        let mask = NSBitmapImageRep(data: maskData)!
        try check(rgba(mask, 32, 16).redComponent > 0.99 && rgba(mask, 32, 16).blueComponent > 0.99,
                  "Mask ellipse must be filled white, not outlined or colored")
        try check(rgba(mask, 0, 0).redComponent < 0.01 && rgba(mask, 0, 0).alphaComponent > 0.99,
                  "Mask must have opaque black background even over transparent source pixels")
        let brushMask = NSBitmapImageRep(data: try Qwen21AnnotationRenderer.render(source: fixture, strokes: [dot], output: .separateMask))!
        try check(rgba(brushMask, 48, 8).redComponent > 0.8 && rgba(brushMask, 48, 24).redComponent < 0.01,
                  "Mask brush coordinate or white ink incorrect")
        let orientedMask = NSBitmapImageRep(data: try Qwen21AnnotationRenderer.render(source: rotated, strokes: [dot], output: .separateMask))!
        try check(orientedMask.pixelsWide == 32 && orientedMask.pixelsHigh == 64, "Mask lost EXIF orientation")
        for invalid in [[], [Qwen21AnnotationStroke(tool: .ellipse, points: [.zero])],
                        [Qwen21AnnotationStroke(tool: .brush, points: [CGPoint(x: -1, y: 0)])]] as [[Qwen21AnnotationStroke]] {
            var rejected = false
            do { _ = try Qwen21AnnotationRenderer.render(source: fixture, strokes: invalid) } catch { rejected = true }
            try check(rejected, "Invalid annotation was accepted")
        }
        let annotatedOK = await studio.annotateQwen21Asset(original[0].id, strokes: [ellipse])
        try check(annotatedOK && studio.draft.assets.count == 2 && studio.draft.assets[0].path != original[0].path,
                  "Annotation did not replace its reference binding with a new copy")
        try check(try Data(contentsOf: fixture) == data, "Annotation overwrote the original file")
        try check(studio.draft.assets[1] == original[1], "Annotation reordered other references")
        studio.undoAssetChange()
        try check(studio.draft.assets == Array(original.prefix(2)), "Annotation binding undo failed")
        let maskPrompt = studio.draft.prompt
        let maskOK = await studio.annotateQwen21Asset(original[0].id, strokes: [ellipse], output: .separateMask)
        try check(maskOK && studio.draft.assets.count == 3, "Separate mask was not appended")
        try check(Array(studio.draft.assets.prefix(2)) == Array(original.prefix(2)) && studio.draft.prompt == maskPrompt,
                  "Separate mask changed existing image indices or prompt")
        try check(studio.message?.contains("<image3>") == true && studio.message?.contains("<image1>") == true,
                  "Mask did not explain source/mask image indices")
        let maskRequest = try studio.draft.request(output: output)
        try check(maskRequest.inputs?.map(\.path) == studio.draft.assets.map(\.path), "Mask absent from native request")
        _ = try NativeEngine.plan(maskRequest)
        try check(try Data(contentsOf: URL(fileURLWithPath: studio.draft.assets[2].path)) == maskData,
                  "Importer changed mask pixels")
        studio.undoAssetChange()
        try check(studio.draft.assets == Array(original.prefix(2)), "Separate mask undo failed")
        studio.draft.assets = original
        let overflowMask = await studio.annotateQwen21Asset(original[0].id, strokes: [ellipse], output: .separateMask)
        try check(!overflowMask && studio.draft.assets == original, "Mask bypassed ten-reference limit")
        studio.draft.assets = Array(original.prefix(2))
        for example in Qwen21PromptExample.allCases {
            studio.applyQwen21Example(example)
            let r = try studio.draft.request(output: output)
            try check(r.prompt == example.prompt && r.execution == "gpu" && r.steps == 40 &&
                      r.width == 512 && r.height == 512,
                      "Example request did not use its visible prompt/GPU/steps")
            try check(r.operation == (example.referenceCount == 0 ? "image.generate" : "image.edit"),
                      "Example selected the wrong operation")
            if example.referenceCount == 0 {
                try check(r.inputs?.isEmpty != false, "Transparent text-to-image forwarded stored references")
            } else {
                try check(r.inputs?.map(\.path) == Array(original.prefix(2)).map(\.path), "Example reordered reference/mask")
            }
            _ = try NativeEngine.plan(r)
        }
        studio.selectModel("flux2-klein-4b")
        try check(studio.imageImportLimit == 8, "Qwen21 change altered existing eight-image models")
        studio.draft.assets = Array(original.prefix(8))
        await studio.addFiles([fixture])
        try check(studio.draft.assets.count == 8, "Existing model overflow behavior changed")
        print("PASS Qwen21 ten-reference import/request contracts, annotation/mask rendering/orientation/alpha/copy/order/undo/limits, prompt examples (no inference quality)")
    }

    @MainActor static func verifyDiTCacheAndOrdinaryLoRA(root: URL, fixture: URL, assets: [StudioAsset]) throws {
        func check(_ ok: @autoclosure () throws -> Bool, _ message: String) throws {
            if try !ok() { throw NativeFailure(message: message) }
        }
        func rejects(_ draft: StudioDraft, _ message: String) throws {
            do { _ = try draft.request(output: root.appendingPathComponent("unused-cache.png")) }
            catch { return }
            throw NativeFailure(message: message)
        }
        let stateRoot = root.appendingPathComponent("dit-cache-state")
        let studio = StudioState(directory: stateRoot)
        studio.selectModel("qwen-image-2.1")
        studio.draft.modelPaths[studio.draft.modelID] = root.path
        studio.draft.prompt = "A ceramic teapot."
        studio.draft.steps = 25
        let base = studio.draft, output = root.appendingPathComponent("unused-cache.png")
        try check(base.qwen21DiTCache == "off" && base.qwen21DiTCacheUnavailableReason == nil, "Cache default or base capability changed")
        for mode in Qwen21DiTCacheMode.allCases {
            var draft = base; draft.qwen21DiTCache = mode.rawValue
            let request = try draft.request(output: output)
            let data = try JSONEncoder().encode(request)
            let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            try check(json["qwen21_dit_cache"] as? String == mode.rawValue, "Legacy request dropped selected cache mode")
            let decoded = try JSONDecoder().decode(NativeRequest.self, from: data)
            try check(decoded.qwen21_dit_cache == mode.rawValue, "Request cache roundtrip failed")
            try check((request.allow_approximation == true) == (mode != .off), "Approximation opt-in differs from cache selection")
            let v2 = NativeRequestV2(legacy: request)
            let v2JSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(v2)) as! [String: Any]
            try check((v2JSON["execution"] as? [String: Any])?["qwen21_dit_cache"] as? String == mode.rawValue,
                      "V2 execution dropped cache mode")
            for planData in [try NativeEngine.plan(request), try NativeEngine.plan(v2)] {
                let plan = try JSONSerialization.jsonObject(with: planData) as! [String: Any]
                try check(plan["qwen21_dit_cache"] as? String == mode.rawValue, "Native plan lost explicit DiT cache mode")
            }
            let restored = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
            try check(restored.qwen21DiTCache == mode.rawValue && restored.steps == 25, "Draft roundtrip lost mode or silently changed steps")
        }
        var cached = base; cached.qwen21DiTCache = "balanced"
        for count in 1...3 {
            var edit = cached; edit.operation = "image.edit"; edit.assets = Array(assets.prefix(count))
            try check(try edit.request(output: output).inputs?.map(\.path) == edit.assets.map(\.path), "Cached editing changed ordered references")
        }
        let invalid: [(String, (inout StudioDraft) -> Void)] = [
            ("low steps", { $0.steps = 19 }), ("high steps", { $0.steps = 41 }),
            ("canvas", { $0.width = 1024 }),
            ("ANE", { $0.acceleration = StudioAcceleration(policy: "gpu_ane") }),
            ("profile", { $0.profilePath = "/tmp/profile.json" }),
            ("PE", { $0.promptEnhance = true }), ("upscale operation", { $0.operation = "image.upscale" }),
            ("references", { $0.operation = "image.edit"; $0.assets = Array(assets.prefix(4)) })
        ]
        for (name, mutate) in invalid {
            var draft = cached; mutate(&draft)
            try check(draft.qwen21DiTCacheUnavailableReason != nil, "Unavailable cache capability not reported: \(name)")
            try rejects(draft, "Invalid cache request accepted: \(name)")
        }
        var followedByUpscale = cached; followedByUpscale.upscaleAfterGeneration = true
        try check(followedByUpscale.qwen21DiTCacheUnavailableReason == nil, "Sequential post-generation upscale incorrectly disables DiT cache")
        let upscaleGeneration = try followedByUpscale.request(output: output)
        try check(upscaleGeneration.operation == "image.generate" && upscaleGeneration.qwen21_dit_cache == "balanced" &&
                  upscaleGeneration.width == 512 && upscaleGeneration.height == 512,
                  "Post-generation upscale changed the preceding Qwen request or dropped its cache mode")
        var unknown = base; unknown.qwen21DiTCache = "future-mode"
        try rejects(unknown, "Unknown cache mode silently normalized")
        var legacyFields = try JSONSerialization.jsonObject(with: JSONEncoder().encode(cached)) as! [String: Any]
        legacyFields.removeValue(forKey: "qwen21DiTCache")
        let legacyDraft = try JSONDecoder().decode(StudioDraft.self, from: JSONSerialization.data(withJSONObject: legacyFields))
        try check(legacyDraft.qwen21DiTCache == "off" && legacyDraft.steps == 25, "Old draft enabled approximation or lost its schedule")

        let ordinary = root.appendingPathComponent("ordinary-style-lora.safetensors")
        try Data([0]).write(to: ordinary)
        studio.draft = cached; studio.addLoRA(ordinary.path)
        try check(studio.draft.qwen21TurboLoRA == nil && studio.draft.steps == 25 && studio.draft.qwen21DiTCache == "balanced" &&
                  studio.draft.loras[0].strength == 1, "Ordinary LoRA was converted into Turbo or lost explicit cache selection")
        try check(studio.imageImportLimit == 3, "Ordinary LoRA exposes more references than its request contract")
        for steps in [20, 25, 40] {
            for strength in [-8.0, 1.0, 8.0] {
                var draft = studio.draft; draft.steps = steps; draft.loras[0].strength = strength
                let request = try draft.request(output: output)
                try check(request.steps == steps && request.loras?.first?.strength == strength && request.lora_strategy == "inference_time",
                          "Ordinary LoRA schedule/strength/strategy lost")
                _ = try NativeEngine.plan(request)
            }
        }
        var invalidLoRA = studio.draft; invalidLoRA.steps = 6; invalidLoRA.qwen21DiTCache = "off"
        try rejects(invalidLoRA, "Ordinary LoRA accepted a Turbo six-step schedule")
        invalidLoRA = studio.draft; invalidLoRA.loras[0].strength = 8.01
        try rejects(invalidLoRA, "LoRA strength exceeds native contract")
        invalidLoRA = studio.draft; invalidLoRA.loras.append(StudioLoRA(path: ordinary.path))
        try check(invalidLoRA.qwen21DiTCacheUnavailableReason != nil, "Cache picker remains available for multiple adapters")
        try rejects(invalidLoRA, "Multiple ordinary adapters accepted")
        invalidLoRA = studio.draft; invalidLoRA.loras[0].role = "text_encoder"
        try check(invalidLoRA.qwen21DiTCacheUnavailableReason != nil, "Cache picker accepts unsupported LoRA role")
        try rejects(invalidLoRA, "Ordinary Qwen LoRA accepted text-encoder role")
        invalidLoRA = studio.draft; invalidLoRA.loraStrategy = "in_memory_merge"
        try check(invalidLoRA.qwen21DiTCacheUnavailableReason != nil, "Cache picker accepts unsupported merge strategy")
        try rejects(invalidLoRA, "Ordinary Qwen LoRA accepted merge strategy")
        for rank in [128, 256] {
            let file = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r\(rank).safetensors")
            try Data([0]).write(to: file)
            var turbo = cached; turbo.steps = 6; turbo.loras = [StudioLoRA(path: file.path)]
            try check(turbo.qwen21TurboLoRA != nil && turbo.qwen21DiTCacheUnavailableReason != nil, "Known Viggle not distinguished from ordinary LoRA")
            try rejects(turbo, "Viggle accepted DiT cache")
            turbo.qwen21DiTCache = "off"
            try check(try turbo.request(output: output).steps == 6, "Existing GPU Turbo contract changed")
            turbo.loras.append(StudioLoRA(path: ordinary.path))
            try rejects(turbo, "Viggle mixed with ordinary LoRA")
        }
        studio.save()
        let reloaded = StudioState(directory: stateRoot)
        try check(reloaded.draft.qwen21DiTCache == "balanced", "Persisted cache mode was not loaded")
        let configuration = root.appendingPathComponent("cache-configuration.json")
        try JSONEncoder().encode(studio.draft).write(to: configuration)
        studio.draft.qwen21DiTCache = "off"
        try studio.importConfiguration(from: configuration)
        try check(studio.draft.qwen21DiTCache == "balanced" && studio.draft.steps == 25, "Configuration import lost explicit cache mode or schedule")
        let request = try studio.draft.request(output: fixture)
        let job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "complete",
                            completed: 25, total: 25, elapsed: 1, modelPath: root.path)
        let savedJob = try JSONDecoder().decode(NativeJob.self, from: JSONEncoder().encode(job))
        studio.selectModel("flux2-klein-4b")
        try check(studio.draft.qwen21DiTCache == "off", "Changing models retained Qwen cache mode")
        studio.draft.modelPaths[studio.draft.modelID] = root.path
        let fluxRequest = try studio.draft.request(output: output)
        let fluxFields = try JSONSerialization.jsonObject(with: JSONEncoder().encode(fluxRequest)) as! [String: Any]
        try check(fluxFields["qwen21_dit_cache"] == nil, "Unrelated model serialized Qwen-specific cache field")
        studio.reuse(savedJob)
        try check(studio.draft.qwen21DiTCache == "balanced" && studio.draft.steps == 25 && studio.draft.qwen21TurboLoRA == nil,
                  "History reuse lost cache mode or converted ordinary LoRA to Turbo")
        var oldRequestFields = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
        oldRequestFields.removeValue(forKey: "qwen21_dit_cache")
        let oldRequest = try JSONDecoder().decode(NativeRequest.self, from: JSONSerialization.data(withJSONObject: oldRequestFields))
        let oldJob = NativeJob(id: UUID(), createdAt: Date(), request: oldRequest, state: "succeeded", phase: "complete",
                               completed: 25, total: 25, elapsed: 1, modelPath: root.path)
        studio.reuse(oldJob)
        try check(studio.draft.qwen21DiTCache == "off", "Old history reused stale cache approximation")
        print("PASS Qwen DiT mode serialization/persistence/history/capabilities and ordinary-vs-Viggle LoRA (CPU only)")
    }

    @MainActor static func verifyTurboAndEditing(root: URL, fixture: URL, assets: [StudioAsset]) async throws {
        func check(_ condition: @autoclosure () throws -> Bool, _ reason: String) throws {
            if try !condition() { throw NativeFailure(message: reason) }
        }
        func rejects(_ work: () throws -> Void, _ reason: String) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: reason)
        }
        let studio = StudioState(directory: root.appendingPathComponent("turbo-state"))
        studio.selectModel("qwen-image-2.1")
        studio.draft.modelPaths[studio.draft.modelID] = root.path
        studio.draft.assets = assets
        let output = root.appendingPathComponent("turbo-unused.png")
        for rank in [128, 256] {
            let adapter = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r\(rank).safetensors")
            // Plans verify the request contract only, never load these bytes.
            try Data([0]).write(to: adapter)
            studio.draft.loras = [StudioLoRA(path: adapter.path, strength: 0.5, role: "text_encoder")]
            studio.applyQwen21TurboPreset()
            try check(studio.draft.steps == 6 && studio.draft.width == 512 && studio.draft.height == 512 &&
                      studio.draft.loras[0].strength == 1 && studio.draft.loras[0].role == "transformer",
                      "Qwen six-step preset failed for rank \(rank)")
            studio.draft.operation = "image.generate"
            studio.draft.acceleration = StudioAcceleration(policy: "gpu_ane")
            let text = try studio.draft.request(output: output)
            try check(text.execution == "gpu" && text.ane_manifest == nil && text.profile == nil &&
                      text.allow_approximation == true && text.lora_strategy == "inference_time" && text.inputs?.isEmpty != false,
                      "Qwen Turbo request lost GPU/approximation/strategy or used stored inputs")
            _ = try NativeEngine.plan(text)
            studio.changeOperation("image.edit")
            for count in 1...3 {
                studio.draft.assets = Array(assets.prefix(count))
                let edit = try studio.draft.request(output: output)
                try check(edit.inputs?.map(\.path) == Array(assets.prefix(count)).map(\.path) &&
                          edit.inputs?.allSatisfy { $0.role == "reference" && $0.strength == nil } == true,
                          "Qwen Turbo lost ordered \(count)-reference editing inputs")
                _ = try NativeEngine.plan(edit)
            }
            studio.draft.assets = Array(assets.prefix(4))
            try rejects({ _ = try studio.draft.request(output: output) }, "Four-reference Turbo edit accepted")
            studio.draft.assets = Array(assets.prefix(3))
            studio.applyQwen21Example(.rgbaEdit)
            try check(try studio.draft.request(output: output).steps == 6,
                      "Applying a prompt example changed the active Turbo adapter to the base schedule")
            studio.draft.loras[0].strength = 0.8
            try rejects({ _ = try studio.draft.request(output: output) }, "Unsupported Turbo strength accepted")
            studio.applyQwen21TurboPreset()
            studio.draft.steps = 40
            try rejects({ _ = try studio.draft.request(output: output) }, "Base schedule accepted for Turbo LoRA")
            studio.applyQwen21TurboPreset()
            studio.draft.loras.append(StudioLoRA(path: adapter.path))
            try rejects({ _ = try studio.draft.request(output: output) }, "Multiple Turbo adapters accepted")
            studio.draft.loras.removeLast()
            let request = try studio.draft.request(output: output)
            let job = NativeJob(id: UUID(), createdAt: Date(), request: request, state: "succeeded", phase: "complete",
                                completed: 6, total: 6, elapsed: 1, modelPath: root.path)
            studio.draft.streaming.selection = .tier20
            studio.draft.ltxBackend = "cpp_mlx"; studio.draft.ltxAccelerationMode = "fast_approx"
            studio.reuse(job)
            let reused = try studio.draft.request(output: output)
            try check(studio.draft.streaming.selection == .off && studio.draft.ltxBackend == "auto" &&
                      studio.draft.ltxAccelerationMode == "quality" && reused.inputs?.map(\.path) == request.inputs?.map(\.path) &&
                      reused.loras?.map(\.path) == request.loras?.map(\.path),
                      "History reuse leaked streaming/LTX state or lost Qwen inputs/LoRA")
            try check(studio.draft.assets.allSatisfy { $0.width == 64 && $0.height == 32 },
                      "History reuse lost reference geometry needed for correctly aligned annotations")
            studio.draft.loras[0].enabled = false
            studio.draft.acceleration = StudioAcceleration(policy: "gpu")
            let disabled = try studio.draft.request(output: output)
            try check(disabled.loras == nil && disabled.allow_approximation != true && disabled.lora_strategy == "auto",
                      "Disabled Turbo LoRA still changed the request")
            studio.draft.loras = []
            _ = try studio.draft.request(output: output)
        }
        var historyRequest = try studio.draft.request(output: output)
        historyRequest.loras = [NativeLoRA(path: root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors").path)]
        let historyJob = NativeJob(id: UUID(), createdAt: Date(), request: historyRequest, state: "succeeded", phase: "complete",
                                   completed: 6, total: 6, elapsed: 1, modelPath: root.path)
        studio.selectModel("flux2-klein-4b")
        let fluxLoRA = StudioLoRA(path: fixture.path, strength: 0.6, enabled: false)
        studio.draft.loras = [fluxLoRA]
        studio.reuse(historyJob)
        studio.selectModel("flux2-klein-4b")
        try check(studio.draft.loras == [fluxLoRA], "History reuse discarded the outgoing model's LoRA selection")
        studio.selectModel("qwen-image-2.1")
        try check(studio.draft.loras.map(\.path) == historyRequest.loras?.map(\.path),
                  "Switching models lost the adapter restored from history")
        let streamed = NativeJob(id: UUID(), createdAt: Date(), request: NativeRequest(prompt: "streaming history", output: output.path),
                                 state: "succeeded", phase: "complete", completed: 4, total: 4, elapsed: 1,
                                 publicStreamingTargetBytes: 16 << 30)
        studio.reuse(streamed)
        try check(studio.draft.streaming.selection == .tier16 && studio.draft.streaming.userSelected &&
                  studio.draft.streaming.status == "unknown" && studio.draft.streaming.requestDigest == nil,
                  "History did not restore streaming intent without reusing old authority")
        var ltxRequest = NativeRequest(prompt: "LTX history", output: output.path)
        ltxRequest.model = "ltx-2.5-distilled"; ltxRequest.operation = "video.generate"
        ltxRequest.steps = 11; ltxRequest.frames = 97; ltxRequest.fps = 24
        ltxRequest.ltx_backend = "c_metal"; ltxRequest.ltx_fast_av = false
        ltxRequest.ltx_sol_stage2 = true; ltxRequest.ltx_stage2_text_rows = 256
        let ltxJob = NativeJob(id: UUID(), createdAt: Date(), request: ltxRequest, state: "succeeded", phase: "complete",
                               completed: 11, total: 11, elapsed: 1)
        studio.reuse(ltxJob)
        try check(studio.draft.ltxBackend == "c_metal" && !studio.draft.ltxFastAV &&
                  studio.draft.ltxAccelerationMode == "fast_approx", "History did not restore LTX execution controls")
        studio.reuse(historyJob)
        try check(studio.draft.streaming.selection == .off && studio.draft.ltxAccelerationMode == "quality",
                  "Qwen history retained unrelated streaming/LTX execution controls")
        studio.draft.loras = []
        studio.draft.assets = assets
        studio.useOnlyAssetForEditing(assets[2].id)
        try check(studio.draft.modelID == "qwen-image-2.1" && studio.draft.operation == "image.edit" &&
                  studio.draft.assets == [assets[2]], "Selecting one Qwen image switched to another model")
        studio.undoAssetChange()
        try check(studio.draft.assets == assets, "Selecting one Qwen image cannot restore references")
        var resultRequest = try studio.draft.request(output: fixture)
        resultRequest.operation = "image.generate"; resultRequest.inputs = []
        let result = NativeJob(id: UUID(), createdAt: Date(), request: resultRequest, state: "succeeded", phase: "complete",
                               completed: 6, total: 6, elapsed: 1, modelPath: root.path)
        let imported = await studio.editResult(result)
        try check(imported && studio.draft.modelID == "qwen-image-2.1" && studio.draft.operation == "image.edit" &&
                  studio.draft.assets.count == 1 && studio.draft.assets[0].path != fixture.path,
                  "Edit result failed with a full input tray or switched Qwen to FLUX")
        try check(try Data(contentsOf: URL(fileURLWithPath: studio.draft.assets[0].path)) == Data(contentsOf: fixture),
                  "Edit result changed source image bytes")
        studio.undoAssetChange()
        try check(studio.draft.assets == assets, "Edit result cannot restore old references")
        var missingRequest = resultRequest; missingRequest.output = root.appendingPathComponent("missing-result.png").path
        let missing = NativeJob(id: UUID(), createdAt: Date(), request: missingRequest, state: "succeeded", phase: "complete",
                                completed: 6, total: 6, elapsed: 1)
        let rejectedMissing = await studio.editResult(missing)
        try check(!rejectedMissing && studio.draft.assets == assets, "Missing result changed the editing draft")
    }
}
