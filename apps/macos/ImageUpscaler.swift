import Foundation
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum UpscaleVariant: String, Codable, CaseIterable, Sendable {
    case x2plus, x4plus
    var scale: Int { self == .x2plus ? 2 : 4 }
    var modelID: String { "real-esrgan-\(rawValue)" }
    static func from(modelID: String) -> Self? { allCases.first { $0.modelID == modelID } }
    var title: String { "Real-ESRGAN \(rawValue) · \(scale)×" }
    var downloadURL: URL { URL(string: "https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_\(rawValue)_522_fp16.zip")! }
    static let processingURL = URL(string: "https://github.com/hanxiao/real-esrgan-coreml#usage")!
}

enum UpscaleCompute: String, Codable, CaseIterable, Sendable {
    case gpu, ane
    var title: String { self == .gpu ? "GPU" : "ANE 优先（可回退 CPU）" }
    var units: MLComputeUnits { self == .gpu ? .cpuAndGPU : .cpuAndNeuralEngine }
}

/// RGB float32 NCHW tile buffers, 2x/4x output; compatible with RealESRGAN_x2plus/x4plus Core ML exports.
/// The App only opens explicitly selected local models and never downloads weights.
protocol UpscalePredictor {
    var size: Int { get }
    var scale: Int { get }
    func predict(_ input: MLMultiArray) throws -> MLMultiArray
}

extension UpscalePredictor { var scale: Int { 4 } }

final class CoreMLUpscalePredictor: UpscalePredictor {
    let size: Int
    let scale: Int
    private let model: MLModel
    private let inputName: String
    private let inputType: MLMultiArrayDataType
    private let outputName: String
    private let compiled: URL?

    init(url: URL, compute: UpscaleCompute = .gpu) throws {
        try ImageUpscaler.validateModelURL(url)
        let temporary = url.pathExtension.lowercased() == "mlmodelc" ? nil : try MLModel.compileModel(at: url)
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = compute.units
            let model = try MLModel(contentsOf: temporary ?? url, configuration: configuration)
            let inputs = model.modelDescription.inputDescriptionsByName
            let outputs = model.modelDescription.outputDescriptionsByName
            guard inputs.count == 1, outputs.count == 1,
                  let input = inputs.first, let output = outputs.first,
                  let ic = input.value.multiArrayConstraint, let oc = output.value.multiArrayConstraint,
                  [.float32, .float16].contains(ic.dataType), [.float32, .float16].contains(oc.dataType) else {
                throw NativeFailure(message: "请选择 x2plus / x4plus 的 Core ML RGB Float16/Float32 NCHW 模型；此模型接口不兼容。")
            }
            let shape = ic.shape.map(\.intValue), out = oc.shape.map(\.intValue)
            guard shape.count == 4, shape[0] == 1, shape[1] == 3,
                  shape[2] == shape[3], (64...1024).contains(shape[2]),
                  out.count == 4, out[0] == 1, out[1] == 3, out[2] == out[3],
                  out[2] % shape[2] == 0, [2, 4].contains(out[2] / shape[2]) else {
                throw NativeFailure(message: "超分模型需要固定 [1,3,N,N] 输入和 2 / 4 倍 RGB 输出（例如 N=522）。")
            }
            self.model = model; size = shape[2]; scale = out[2] / shape[2]
            inputName = input.key; inputType = ic.dataType; outputName = output.key; compiled = temporary
        } catch {
            if let temporary { try? FileManager.default.removeItem(at: temporary) }
            throw error
        }
    }
    deinit { if let compiled { try? FileManager.default.removeItem(at: compiled) } }
    func predict(_ input: MLMultiArray) throws -> MLMultiArray {
        let tensor: MLMultiArray
        if inputType == .float16 {
            tensor = try MLMultiArray(shape: input.shape, dataType: .float16)
            let src = input.dataPointer.assumingMemoryBound(to: Float.self)
            let dst = tensor.dataPointer.bindMemory(to: Float16.self, capacity: tensor.count)
            for i in 0..<tensor.count { dst[i] = Float16(src[i]) }
        } else { tensor = input }
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: tensor]))
        guard let value = result.featureValue(for: outputName)?.multiArrayValue else {
            throw NativeFailure(message: "超分模型未返回 RGB 张量。")
        }
        return value
    }
}

enum ImageUpscaler {
    static let maximumOutputPixels = 64 * 1024 * 1024
    static func validateModelURL(_ url: URL) throws {
        guard ["mlpackage", "mlmodelc", "mlmodel"].contains(url.pathExtension.lowercased()),
              FileManager.default.fileExists(atPath: url.path) else {
            throw NativeFailure(message: "请先选择本地 Real-ESRGAN x2plus / x4plus Core ML 模型（.mlpackage / .mlmodelc / .mlmodel）；不会自动下载模型。")
        }
    }
    static func dimensions(_ url: URL, scale: Int = 4) throws -> (width: Int, height: Int) {
        guard [2, 4].contains(scale), let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              CGImageSourceGetCount(source) == 1,
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0, w <= 8192, h <= 8192, w * h <= maximumOutputPixels / (scale * scale) else {
            throw NativeFailure(message: "请选择单张静态图片；超分后的像素数不能超过 64 Mi 像素。原图不会被修改。")
        }
        let orientation = props[kCGImagePropertyOrientation] as? Int ?? 1
        return (5...8).contains(orientation) ? (h, w) : (w, h)
    }
    static func reflect(_ index: Int, length: Int) -> Int {
        guard length > 1 else { return 0 }
        let period = 2 * (length - 1), value = ((index % period) + period) % period
        return value < length ? value : period - value
    }
    static func starts(length: Int, tile: Int, overlap: Int) -> [Int] {
        if length <= tile { return [0] }
        var result = [0]
        while result.last! + tile < length {
            result.append(min(result.last! + tile - overlap, length - tile))
        }
        return result
    }
    static func run(source: URL, destination: URL, model: URL, compute: UpscaleCompute = .gpu,
                    progress: @Sendable (Int, Int) -> Void) throws {
        try Task.checkCancellation()
        let predictor = try CoreMLUpscalePredictor(url: model, compute: compute)
        try render(source: source, destination: destination, predictor: predictor, progress: progress)
    }
    // Separate tensor plumbing from model loading so geometry, padding, alpha and
    // cancellation can be checked without downloading or pretending to have weights.
    static func render(source: URL, destination: URL, predictor: any UpscalePredictor,
                       progress: @Sendable (Int, Int) -> Void) throws {
        let scale = predictor.scale
        let dimensions = try dimensions(source, scale: scale)
        let w = dimensions.width, h = dimensions.height, ow = w * scale, oh = h * scale
        guard (64...1024).contains(predictor.size) else { throw NativeFailure(message: "无效的超分模型尺寸。") }
        guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: max(w, h)
              ] as CFDictionary), image.width == w, image.height == h else {
            throw NativeFailure(message: "无法解码超分原图。")
        }
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        try rgba.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4, space: space, bitmapInfo: info) else {
                throw NativeFailure(message: "无法创建超分图像缓冲区。")
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        try Task.checkCancellation()
        let n = predictor.size, tile = n - 10, overlap = min(32, tile / 2)
        let xs = starts(length: w, tile: tile, overlap: overlap), ys = starts(length: h, tile: tile, overlap: overlap)
        var rgb = [Float](repeating: 0, count: ow * oh * 3)
        var weights = [Float](repeating: 0, count: ow * oh)
        var done = 0
        progress(0, xs.count * ys.count)
        for y in ys { for x in xs {
            try Task.checkCancellation()
            try autoreleasepool {
                let tw = min(tile, w - x), th = min(tile, h - y)
                let input = try MLMultiArray(shape: [1, 3, NSNumber(value: n), NSNumber(value: n)], dataType: .float32)
                let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: 3 * n * n)
                for iy in 0..<n { for ix in 0..<n {
                    let i = ((y + reflect(iy, length: th)) * w + x + reflect(ix, length: tw)) * 4
                    let alpha = Float(rgba[i + 3])
                    for c in 0..<3 { ptr[c * n * n + iy * n + ix] = alpha > 0 ? Float(rgba[i + c]) / alpha : 0 }
                } }
                let output = try predictor.predict(input)
                guard output.shape.map(\.intValue) == [1, 3, n * scale, n * scale],
                      [.float32, .float16].contains(output.dataType) else {
                    throw NativeFailure(message: "超分输出尺寸或类型与模型的 RGB 放大倍数不匹配。")
                }
                let strides = output.strides.map(\.intValue)
                let f32 = output.dataPointer.assumingMemoryBound(to: Float.self)
                let f16 = output.dataPointer.assumingMemoryBound(to: Float16.self)
                for iy in 0..<th * scale {
                    try Task.checkCancellation()
                    for ix in 0..<tw * scale {
                        let ramp = Float(overlap * scale)
                        var weight: Float = 1
                        if x > 0 { weight *= min(1, Float(ix + 1) / ramp) }
                        if y > 0 { weight *= min(1, Float(iy + 1) / ramp) }
                        if x + tw < w { weight *= min(1, Float(tw * scale - ix) / ramp) }
                        if y + th < h { weight *= min(1, Float(th * scale - iy) / ramp) }
                        let pixel = (y * scale + iy) * ow + x * scale + ix
                        weights[pixel] += weight
                        for c in 0..<3 {
                            let index = c * strides[1] + iy * strides[2] + ix * strides[3]
                            let value = output.dataType == .float32 ? f32[index] : Float(f16[index])
                            guard value.isFinite else { throw NativeFailure(message: "超分模型返回了无效像素。") }
                            rgb[pixel * 3 + c] += min(1, max(0, value)) * weight
                        }
                    }
                }
            }
            done += 1; progress(done, xs.count * ys.count)
        } }
        var result = [UInt8](repeating: 0, count: ow * oh * 4)
        // Alpha is resampled separately; the RGB network cannot invent transparency.
        for y in 0..<oh {
            try Task.checkCancellation()
            let sy = max(0, min(Float(h - 1), (Float(y) + 0.5) / Float(scale) - 0.5)), y0 = Int(sy), fy = sy - Float(y0)
            for x in 0..<ow {
                let sx = max(0, min(Float(w - 1), (Float(x) + 0.5) / Float(scale) - 0.5)), x0 = Int(sx), fx = sx - Float(x0)
                func a(_ ix: Int, _ iy: Int) -> Float { Float(rgba[(iy * w + ix) * 4 + 3]) }
                let alpha = (a(x0, y0) * (1 - fx) + a(min(w - 1, x0 + 1), y0) * fx) * (1 - fy)
                    + (a(x0, min(h - 1, y0 + 1)) * (1 - fx) + a(min(w - 1, x0 + 1), min(h - 1, y0 + 1)) * fx) * fy
                let i = y * ow + x
                for c in 0..<3 { result[i * 4 + c] = UInt8(min(255, max(0, (rgb[i * 3 + c] / weights[i] * alpha).rounded()))) }
                result[i * 4 + 3] = UInt8(alpha.rounded())
            }
        }
        try Task.checkCancellation()
        guard let provider = CGDataProvider(data: Data(result) as CFData),
              let image = CGImage(width: ow, height: oh, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: ow * 4, space: space, bitmapInfo: CGBitmapInfo(rawValue: info),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let writer = CGImageDestinationCreateWithURL(destination as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw NativeFailure(message: "无法保存超分结果。")
        }
        CGImageDestinationAddImage(writer, image, nil)
        guard CGImageDestinationFinalize(writer) else { throw NativeFailure(message: "保存超分结果失败。") }
        try Task.checkCancellation()
    }
}

struct UpscaleModelInfo: Sendable, Equatable {
    let identity: String
    let path: String
    let compute: UpscaleCompute
    let scale: Int
    let inputSize: Int
    var cacheHit: Bool
}

/// A single bounded resident model. All Core ML work is serialized off the UI
/// actor; changing the model or device releases the previous session first.
actor UpscaleSession {
    private var predictor: CoreMLUpscalePredictor?
    private var info: UpscaleModelInfo?

    private func identity(_ url: URL, compute: UpscaleCompute) throws -> String {
        let root = url.resolvingSymlinksInPath().standardizedFileURL
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
        var files = [root]
        if let iterator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys) {
            for case let file as URL in iterator { files.append(file) }
        }
        var signature = [root.path, compute.rawValue]
        for file in files {
            let values = try file.resolvingSymlinksInPath().resourceValues(forKeys: Set(keys))
            if values.isRegularFile == true {
                signature.append("\(file.path):\(values.fileSize ?? 0):\(values.contentModificationDate?.timeIntervalSince1970 ?? 0)")
            }
        }
        return signature.sorted().joined(separator: "\n")
    }
    func prepare(model: URL, compute: UpscaleCompute, warmup: Bool = true) throws -> UpscaleModelInfo {
        try Task.checkCancellation()
        try ImageUpscaler.validateModelURL(model)
        let key = try identity(model, compute: compute)
        if var hit = info, hit.identity == key, predictor != nil { hit.cacheHit = true; return hit }
        predictor = nil; info = nil
        let loaded = try CoreMLUpscalePredictor(url: model, compute: compute)
        try Task.checkCancellation()
        // Preload pays execution-plan startup ahead of time. A requested image
        // itself warms the model, so never add a redundant dummy prediction then.
        if warmup {
            let zeros = try MLMultiArray(shape: [1, 3, NSNumber(value: loaded.size), NSNumber(value: loaded.size)], dataType: .float32)
            zeros.dataPointer.bindMemory(to: Float.self, capacity: zeros.count).initialize(repeating: 0, count: zeros.count)
            _ = try loaded.predict(zeros)
        }
        try Task.checkCancellation()
        let result = UpscaleModelInfo(identity: key, path: model.path, compute: compute,
                                      scale: loaded.scale, inputSize: loaded.size, cacheHit: false)
        predictor = loaded; info = result
        return result
    }
    func render(source: URL, destination: URL, expected: UpscaleModelInfo,
                progress: @Sendable (Int, Int) -> Void) throws {
        let current = try prepare(model: URL(fileURLWithPath: expected.path), compute: expected.compute, warmup: false)
        guard current.identity == expected.identity, let predictor else {
            throw NativeFailure(message: "超分模型在准备后发生了变化，请重新载入。")
        }
        try ImageUpscaler.render(source: source, destination: destination, predictor: predictor, progress: progress)
    }
    func release() { predictor = nil; info = nil }
}
