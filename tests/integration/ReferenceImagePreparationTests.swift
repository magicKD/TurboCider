import AppKit
import Combine
import Foundation
import ImageIO
import UniformTypeIdentifiers

// CPU-only image and draft contracts. This target never loads model weights or
// submits generation. Written during the test freeze; execution is deferred.
@main struct ReferenceImagePreparationTests {
    static func check(_ value: @autoclosure () throws -> Bool, _ reason: String) throws {
        guard try value() else { throw NativeFailure(message: reason) }
    }

    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-reference-preparation-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for preset in ReferenceImagePreparation.allCases {
            let small = try preset.dimensions(width: 64, height: 32)
            try check(small.width == 64 && small.height == 32, "Preset upscaled a small original")
        }
        let automatic = try ReferenceImagePreparation.automatic.dimensions(width: 2048, height: 1024)
        let square = try ReferenceImagePreparation.fit512.dimensions(width: 2048, height: 1024)
        let portrait = try ReferenceImagePreparation.portrait512.dimensions(width: 1024, height: 2048)
        let landscape = try ReferenceImagePreparation.landscape512.dimensions(width: 2048, height: 1024)
        try check(automatic.width == 1024 && automatic.height == 512, "Automatic longest-side bound changed")
        try check(square.width == 512 && square.height == 256, "Square fit cropped or padded the original")
        try check(portrait.width == 384 && portrait.height == 768, "Portrait fit changed aspect ratio")
        try check(landscape.width == 768 && landscape.height == 384, "Landscape fit changed aspect ratio")
        for dimensions in [(0, 32), (32, -1), (9000, 9000)] {
            var rejected = false
            do { _ = try ReferenceImagePreparation.automatic.dimensions(width: dimensions.0, height: dimensions.1) }
            catch { rejected = true }
            try check(rejected, "Invalid or oversized dimensions were accepted")
        }

        let source = root.appendingPathComponent("original.png")
        let image = try fixture(width: 2048, height: 1024)
        try write(image, to: source, type: .png)
        let originalBytes = try Data(contentsOf: source)
        let fitted = try ReferenceImagePreparationRenderer.render(source: source, preset: .fit512)
        guard let fittedPNG = fitted.png, let fittedBitmap = NSBitmapImageRep(data: fittedPNG) else {
            throw NativeFailure(message: "Resizing did not produce a PNG")
        }
        try check(fittedBitmap.pixelsWide == 512 && fittedBitmap.pixelsHigh == 256, "PNG dimensions differ from prediction")
        try check(fittedBitmap.colorAt(x: 8, y: 8)!.alphaComponent < 0.01, "Resizing flattened transparency")
        try check(fittedBitmap.colorAt(x: 256, y: 128)!.alphaComponent > 0.99, "Resizing made opaque pixels transparent")
        let restoration = try ReferenceImagePreparationRenderer.render(source: source, preset: .original)
        try check(restoration.png == nil && restoration.width == 2048 && restoration.height == 1024,
                  "Original selection rewrote bytes or dimensions")
        let smallSource = root.appendingPathComponent("small.png")
        try write(try fixture(width: 64, height: 32), to: smallSource, type: .png)
        let smallBytes = try Data(contentsOf: smallSource)
        let unchanged = try ReferenceImagePreparationRenderer.render(source: smallSource, preset: .fit512)
        try check(unchanged.png == nil && (try Data(contentsOf: smallSource)) == smallBytes,
                  "A no-op fit converted source color/bit-depth or created a derivative")

        let rotated = root.appendingPathComponent("orientation-6.tiff")
        try write(image, to: rotated, type: .tiff, orientation: 6)
        let oriented = try ReferenceImagePreparationRenderer.render(source: rotated, preset: .portrait512)
        try check(oriented.originalWidth == 1024 && oriented.originalHeight == 2048 &&
                  oriented.width == 384 && oriented.height == 768, "EXIF rotation did not precede sizing")
        guard let orientedPNG = oriented.png,
              let orientedSource = CGImageSourceCreateWithData(orientedPNG as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(orientedSource, 0, nil) as? [CFString: Any] else {
            throw NativeFailure(message: "Cannot inspect oriented PNG")
        }
        try check((properties[kCGImagePropertyOrientation] as? Int ?? 1) == 1,
                  "PNG retained EXIF rotation after transforming its pixels")

        let stateDirectory = root.appendingPathComponent("state")
        let studio = StudioState(directory: stateDirectory, models: [])
        let first = try await studio.importer.importFile(source)
        let second = try await studio.importer.importFile(source)
        let originals = [first, second]
        studio.draft.assets = originals; studio.draft.initImageID = second.id
        studio.draft.prompt = "Retain this prompt"; studio.draft.width = 512; studio.draft.height = 768
        let prepared = await studio.prepareAssets(ids: Set(originals.map(\.id)), preset: .fit512)
        try check(prepared && studio.draft.assets.map(\.id) == originals.map(\.id) && studio.draft.initImageID == second.id,
                  "Batch preparation changed logical IDs, order or the selected source")
        try check(studio.draft.prompt == "Retain this prompt" && studio.draft.width == 512 && studio.draft.height == 768,
                  "Input preparation changed prompt or output canvas")
        for asset in studio.draft.assets {
            try check(asset.original != nil && asset.path != asset.originalImage.path &&
                      (try Data(contentsOf: URL(fileURLWithPath: asset.originalImage.path))) == originalBytes,
                      "Preparation overwrote or lost an original copy")
        }
        studio.undoAssetChange()
        try check(studio.draft.assets == originals && studio.draft.initImageID == second.id, "Batch preparation requires multiple undo actions")
        let fitAgain = await studio.prepareAsset(id: first.id, preset: .fit512)
        let automaticAgain = await studio.prepareAsset(id: first.id, preset: .automatic)
        try check(fitAgain && automaticAgain && studio.draft.assets[0].width == 1024 && studio.draft.assets[0].height == 512,
                  "Repeated preparation used the smaller derivative instead of the original")
        let restored = await studio.prepareAsset(id: first.id, preset: .original)
        try check(restored && studio.draft.assets[0].path == first.path && studio.draft.assets[0].width == first.width,
                  "Original restoration did not restore the retained copy")

        // A failing second item must roll back the first staged derivative.
        var missing = second
        missing.original = StudioAssetOriginal(path: root.appendingPathComponent("missing.png").path,
                                               name: second.name, width: second.width, height: second.height)
        studio.draft.assets = [first, missing]
        let beforeFailure = studio.draft.assets
        let filesBeforeFailure = try files(in: stateDirectory.appendingPathComponent("inputs"))
        let failure = await studio.prepareAssets(ids: [first.id, missing.id], preset: .fit512)
        try check(!failure && studio.draft.assets == beforeFailure && !studio.importing,
                  "A partially failing batch published changes or kept the busy state")
        try check(try files(in: stateDirectory.appendingPathComponent("inputs")) == filesBeforeFailure,
                  "A failed batch leaked derivatives or removed retained inputs")

        // A context change while the actor is preparing must win over its late result.
        studio.draft.assets = originals
        let contextObserver = studio.$importing.dropFirst().sink { importing in
            if importing { studio.draft.assets.reverse() }
        }
        let conflicted = await studio.prepareAsset(id: first.id, preset: .fit512)
        contextObserver.cancel()
        try check(!conflicted && studio.draft.assets == Array(originals.reversed()), "A stale preparation replaced a newer reference context")
        try check(try files(in: stateDirectory.appendingPathComponent("inputs")) == filesBeforeFailure,
                  "A context conflict left a staged derivative")

        // Old drafts have only the existing asset fields; their current image is
        // a valid original baseline when the new optional metadata is absent.
        let legacyObject: [String: Any] = ["id": first.id.uuidString, "path": first.path,
                                          "name": first.name, "width": first.width, "height": first.height]
        let legacy = try JSONDecoder().decode(StudioAsset.self, from: JSONSerialization.data(withJSONObject: legacyObject))
        try check(legacy.original == nil && legacy.preparation == nil && legacy.originalImage.path == first.path,
                  "Legacy asset metadata was not optional")
        studio.draft.assets = [legacy]
        studio.save()
        let restoredDraft = StudioState(directory: stateDirectory, models: [])
        try check(restoredDraft.draft.assets == [legacy], "Old metadata did not survive draft persistence")
        let migrated = await studio.prepareAsset(id: legacy.id, preset: .automatic)
        try check(migrated && studio.draft.assets[0].originalImage.path == legacy.path, "Preparing an old draft lost its baseline")

        // Annotation is a new pixel baseline, with the same logical reference.
        studio.draft.modelID = "qwen-image-2.1"; studio.draft.operation = "image.edit"
        studio.draft.assets = originals; studio.draft.initImageID = second.id
        let stroke = Qwen21AnnotationStroke(tool: .ellipse,
            points: [CGPoint(x: 0.25, y: 0.25), CGPoint(x: 0.75, y: 0.75)], width: 0.1)
        let annotated = await studio.annotateQwen21Asset(second.id, strokes: [stroke])
        let annotation = studio.draft.assets[1]
        try check(annotated && annotation.id == second.id && annotation.originalImage.path == annotation.path &&
                  annotation.preparation == .original && studio.draft.initImageID == second.id,
                  "Annotation changed its logical selection or inherited an unannotated processing baseline")
        let resizedAnnotation = await studio.prepareAsset(id: second.id, preset: .fit512)
        try check(resizedAnnotation && studio.draft.assets[1].originalImage.path == annotation.path,
                  "Preparing an annotation used the unannotated original")
        let annotationRestored = await studio.prepareAsset(id: second.id, preset: .original)
        try check(annotationRestored && studio.draft.assets[1].path == annotation.path, "Original restoration silently removed annotation ink")
        studio.draft.assets = originals
        studio.reorderAsset(first.id, to: second.id)
        try check(studio.draft.assets == [second, first] && studio.draft.initImageID == second.id,
                  "Drag reorder changed the selected source or missed the target slot")
        studio.undoAssetChange()
        try check(studio.draft.assets == originals, "Drag reorder undo did not restore reference order")
        try check(try Data(contentsOf: source) == originalBytes, "Reference workflow changed the user source file")
        print("PASS reference preparation sizing/orientation/alpha/no-op/source preservation, batch atomicity/context, legacy metadata, annotation baseline and reorder (CPU only)")
    }

    private static func fixture(width: Int, height: Int) throws -> CGImage {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                  bytesPerRow: width * 4, space: space,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
            throw NativeFailure(message: "Cannot allocate fixture")
        }
        context.setFillColor(CGColor(red: 0.2, green: 0.8, blue: 0.2, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        context.clear(CGRect(x: 0, y: 0, width: CGFloat(width / 8), height: CGFloat(height)))
        guard let image = context.makeImage() else { throw NativeFailure(message: "Cannot create fixture") }
        return image
    }

    private static func write(_ image: CGImage, to url: URL, type: UTType, orientation: Int = 1) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw NativeFailure(message: "Cannot create image fixture")
        }
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation] as CFDictionary)
        try check(CGImageDestinationFinalize(destination), "Cannot save image fixture")
    }

    private static func files(in directory: URL) throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
    }
}
