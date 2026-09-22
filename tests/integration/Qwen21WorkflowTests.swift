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
}
