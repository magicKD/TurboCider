import Foundation
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Synthetic tensor plumbing only: this is not a Real-ESRGAN quality test.
private struct NearestPredictor: UpscalePredictor {
    let size = 64
    var scale = 4
    var invalid = false
    func predict(_ input: MLMultiArray) throws -> MLMultiArray {
        let n = size * scale
        let result = try MLMultiArray(shape: [1, 3, NSNumber(value: n), NSNumber(value: n)], dataType: .float32)
        let src = input.dataPointer.assumingMemoryBound(to: Float.self)
        let dst = result.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<3 { for y in 0..<n { for x in 0..<n {
            dst[c * n * n + y * n + x] = invalid ? .nan : src[c * size * size + (y / scale) * size + x / scale]
        } } }
        return result
    }
}
@main struct ImageUpscalerTests {
    @MainActor static func main() async throws {
        func check(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
            guard try value() else { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }; throw NativeFailure(message: "Expected rejection")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-upscale-tests-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.png"), output = root.appendingPathComponent("output.png")
        let w = 79, h = 61
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h { for x in 0..<w {
            let i = (y * w + x) * 4
            rgba[i] = UInt8(x * 3); rgba[i + 1] = UInt8(y * 4); rgba[i + 2] = 31; rgba[i + 3] = 255
        } }
        func write(_ pixels: [UInt8], _ width: Int, _ height: Int, _ url: URL, orientation: Int = 1) throws {
            let space = CGColorSpace(name: CGColorSpace.sRGB)!
            let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: CGDataProvider(data: Data(pixels) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            let writer = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(writer, image, [kCGImagePropertyOrientation: orientation] as CFDictionary)
            try check(CGImageDestinationFinalize(writer), "Fixture export failed")
        }
        func read(_ url: URL) throws -> [UInt8] {
            let image = CGImageSourceCreateImageAtIndex(CGImageSourceCreateWithURL(url as CFURL, nil)!, 0, nil)!
            var data = [UInt8](repeating: 0, count: image.width * image.height * 4)
            data.withUnsafeMutableBytes { buffer in
                let context = CGContext(data: buffer.baseAddress, width: image.width, height: image.height,
                    bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)!
                context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            }
            return data
        }
        try write(rgba, w, h, source)
        let original = try Data(contentsOf: source)
        for scale in [2, 4] {
        try ImageUpscaler.render(source: source, destination: output, predictor: NearestPredictor(scale: scale)) { _, _ in }
        let size = try ImageUpscaler.dimensions(output)
        try check(size.width == w * scale && size.height == h * scale, "Wrong output dimensions")
        let actual = try read(output)
        for y in 0..<h * scale { for x in 0..<w * scale { for c in 0..<4 {
            let expected = rgba[((y / scale) * w + x / scale) * 4 + c]
            try check(abs(Int(actual[(y * w * scale + x) * 4 + c]) - Int(expected)) <= 1, "Tile seam, RGB ordering or orientation mismatch")
        } } }
        }
        try check(Data(contentsOf: source) == original, "Original was modified")
        try check(ImageUpscaler.reflect(17, length: 1) == 0 && ImageUpscaler.reflect(4, length: 3) == 0, "Padding is not reflected")
        let alpha = root.appendingPathComponent("alpha.png"), alphaOut = root.appendingPathComponent("alpha-out.png")
        try write([64, 32, 16, 128], 1, 1, alpha)
        try ImageUpscaler.render(source: alpha, destination: alphaOut, predictor: NearestPredictor()) { _, _ in }
        let transparent = try read(alphaOut)
        try check(transparent.count == 64 && transparent.enumerated().allSatisfy { abs(Int($0.element) - [64, 32, 16, 128][$0.offset % 4]) <= 1 }, "Alpha was lost or multiplied twice")
        let invalid = root.appendingPathComponent("invalid.png")
        try rejects { try ImageUpscaler.render(source: source, destination: invalid, predictor: NearestPredictor(invalid: true)) { _, _ in } }
        try check(!FileManager.default.fileExists(atPath: invalid.path), "Invalid tensor published an image")
        let cancelled = root.appendingPathComponent("cancelled.png")
        let task = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            try ImageUpscaler.render(source: source, destination: cancelled, predictor: NearestPredictor()) { _, _ in }
        }
        do { try await task.value; throw NativeFailure(message: "Cancellation ignored") } catch is CancellationError {}
        try check(!FileManager.default.fileExists(atPath: cancelled.path), "Cancellation published output")
        let missing = root.appendingPathComponent("missing.mlpackage")
        try rejects { try ImageUpscaler.validateModelURL(missing) }
        let store = NativeJobStore(directory: root.appendingPathComponent("store"))
        do { _ = try await store.upscale(source: source, modelURL: missing); throw NativeFailure(message: "Missing model accepted") }
        catch { try check(!store.busy && store.jobs.isEmpty, "Missing model left a job or busy state") }
        var draft = StudioDraft(); draft.upscaleModelPath = missing.path; draft.upscaleAfterGeneration = true; draft.upscaleCompute = .ane
        let restored = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
        try check(restored.upscaleAfterGeneration && restored.upscaleModelPath == missing.path && restored.upscaleCompute == .ane, "Upscale settings did not persist")
        let studio = StudioState(directory: root.appendingPathComponent("draft-store"))
        studio.draft = draft; studio.newDraft()
        try check(studio.draft.upscaleModelPath == missing.path && !studio.draft.upscaleAfterGeneration && studio.draft.upscaleCompute == .ane,
                  "New creation lost the installed upscale model or enabled processing implicitly")
        studio.draft.upscaleModelPaths["x2plus"] = "/local/x2.mlpackage"
        studio.selectGenerationUpscale(.x2plus)
        try check(studio.generationUpscaleVariant == .x2plus && studio.draft.upscaleModelPath == "/local/x2.mlpackage", "2x choice did not select its model")
        studio.selectGenerationUpscale(.x4plus)
        try check(studio.generationUpscaleVariant == .x4plus && studio.draft.upscaleModelPath == missing.path, "4x choice lost its model")
        studio.selectGenerationUpscale(nil)
        let choices = StudioState(directory: root.appendingPathComponent("draft-store"))
        try check(choices.generationUpscaleVariant == nil && choices.draft.upscaleModelPath == missing.path
                  && choices.draft.upscaleModelPaths["x2plus"] == "/local/x2.mlpackage", "Opting out or restart discarded model paths or enabled upscaling")
        let legacy = try JSONDecoder().decode(StudioDraft.self, from: Data("{}".utf8))
        try check(!legacy.upscaleAfterGeneration && legacy.upscaleModelPath.isEmpty && legacy.upscaleCompute == .gpu, "Legacy draft enabled upscaling")
        print("PASS: synthetic 2x/4x RGB/tiling/alpha, input preservation, invalid tensors, cancellation, missing local weights and draft persistence")
        if CommandLine.arguments.count == 4 {
            let model = URL(fileURLWithPath: CommandLine.arguments[1])
            let image = URL(fileURLWithPath: CommandLine.arguments[2])
            let state = URL(fileURLWithPath: CommandLine.arguments[3])
            let live = NativeJobStore(directory: state)
            try await live.preloadUpscaler(modelURL: model, compute: .gpu)
            try check(live.upscaleReady != nil && !live.busy, "Preload did not become ready")
            try await live.preloadUpscaler(modelURL: model, compute: .gpu)
            try check(live.upscaleReady?.cacheHit == true, "Preload did not reuse resident model")
            let job = try await live.upscale(source: image, modelURL: model)
            try check(job.hasOutput && job.request.operation == "image.upscale" && !live.busy, "Real upscaler did not publish a completed job")
            let receipt = try JSONSerialization.jsonObject(with: Data(job.resultJSON!.utf8)) as! [String: Any]
            try check(receipt["model_cache_hit"] as? Bool == true, "Image job ignored preloaded model")
            let restored = NativeJobStore(directory: state)
            try check(restored.jobs.first?.id == job.id && restored.jobs.first?.hasOutput == true, "Real upscale history did not survive restart")
            let work = Task { try await live.upscale(source: image, modelURL: model) }
            while live.activeJob == nil { await Task.yield() }
            live.cancel()
            do { _ = try await work.value; throw NativeFailure(message: "Real upscale cancellation ignored") } catch is CancellationError {}
            try check(!live.busy && live.jobs.first?.state == "cancelled" &&
                      !FileManager.default.fileExists(atPath: live.jobs.first!.request.output), "Real cancellation left output or busy state")
            let gpuIdentity = live.upscaleReady!.identity
            try await live.preloadUpscaler(modelURL: model, compute: .ane)
            try check(live.upscaleReady?.compute == .ane && live.upscaleReady?.cacheHit == false && live.upscaleReady?.identity != gpuIdentity,
                      "Device switch reused the wrong compute configuration")
            await live.releaseUpscaler()
            try check(live.upscaleReady == nil && !live.busy, "Release retained model state")
            print("PASS: real model preload/reuse/release, export, history restart and cancellation; output: \(job.request.output)")
        }

    }
}
