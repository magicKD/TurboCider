import Foundation
import Darwin
import ImageIO
import CoreGraphics
import CryptoKit

/// One image request owns a private staging directory beside its destination.
/// Validation runs off the main actor; publication is the final synchronous step.
final class ImageOutputTransaction: @unchecked Sendable {
    let stagedURL: URL
    private let destination: URL
    private let directory: URL
    private let request: NativeRequest
    private let operation: String
    private var verifiedIdentity: [Int64]?
    private var verifiedFile: FileHandle?
    private var published = false

    private static let crcTable: [UInt32] = (0..<256).map { value in
        var crc = UInt32(value)
        for _ in 0..<8 { crc = (crc >> 1) ^ (crc & 1 == 0 ? 0 : 0xedb88320) }
        return crc
    }

    // ImageIO can recover truncated PNGs. Require complete framing and valid
    // CRCs as well as a successful decode before treating an export as complete.
    private func validatePNG(_ file: FileHandle, bytes: UInt64) throws {
        func invalid() -> NativeFailure {
            NativeFailure(message: "image_artifact_invalid: PNG 数据不完整或校验失败。")
        }
        func read(_ count: Int) throws -> Data {
            var data = Data()
            while data.count < count {
                try Task.checkCancellation()
                guard let part = try file.read(upToCount: count - data.count), !part.isEmpty else { throw invalid() }
                data.append(part)
            }
            return data
        }
        func number(_ data: Data) -> UInt32 {
            data.reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        }
        func update(_ crc: inout UInt32, _ data: Data) {
            for byte in data { crc = Self.crcTable[Int((crc ^ UInt32(byte)) & 255)] ^ (crc >> 8) }
        }
        guard bytes >= 8, try read(8) == Data([137, 80, 78, 71, 13, 10, 26, 10]) else { throw invalid() }
        var remaining = bytes - 8
        var first = true, pixels = false, endedPixels = false
        while remaining >= 12 {
            let header = try read(8)
            let count = UInt64(number(Data(header.prefix(4))))
            let type = Data(header.suffix(4))
            guard count <= remaining - 12 else { throw invalid() }
            if first {
                guard type == Data("IHDR".utf8), count == 13 else { throw invalid() }
                first = false
            } else if type == Data("IHDR".utf8) { throw invalid() }
            if type == Data("IDAT".utf8) {
                guard !endedPixels else { throw invalid() }
                pixels = true
            } else if pixels { endedPixels = true }
            var crc = UInt32.max
            update(&crc, type)
            var payload = count
            while payload > 0 {
                let part = try read(Int(min(payload, 1024 * 1024)))
                update(&crc, part)
                payload -= UInt64(part.count)
            }
            guard try number(read(4)) == crc ^ UInt32.max else { throw invalid() }
            remaining -= count + 12
            if type == Data("IEND".utf8) {
                guard pixels, count == 0, remaining == 0 else { throw invalid() }
                try file.seek(toOffset: 0)
                return
            }
        }
        throw invalid()
    }

    init(request: NativeRequest) throws {
        self.request = request
        operation = NativeRequestV2(legacy: request).operation
        destination = URL(fileURLWithPath: request.output).standardizedFileURL
        guard operation.hasPrefix("image."), destination.pathExtension == "png" else {
            throw NativeFailure(message: "image_transaction_request_invalid: 图片输出必须为 PNG。")
        }
        directory = destination.deletingLastPathComponent()
            .appendingPathComponent(".tc-image-staging-\(UUID())", isDirectory: true)
        stagedURL = directory.appendingPathComponent("output.png")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
    }

    deinit {
        try? verifiedFile?.close()
        try? FileManager.default.removeItem(at: directory)
    }

    private func identity(fd: Int32? = nil) throws -> [Int64] {
        var value = stat()
        let status = fd.map { fstat($0, &value) } ?? lstat(stagedURL.path, &value)
        guard status == 0, value.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), value.st_size > 0 else {
            throw NativeFailure(message: "image_artifact_invalid: 图片为空或不是普通文件。")
        }
        return [Int64(value.st_dev), Int64(bitPattern: value.st_ino), value.st_size,
                Int64(value.st_mtimespec.tv_sec), Int64(value.st_mtimespec.tv_nsec),
                Int64(value.st_ctimespec.tv_sec), Int64(value.st_ctimespec.tv_nsec)]
    }

    func prepare(_ result: Data) throws -> Data {
        struct Summary: Decodable {
            let schema_version: Int
            let model: String
            let operation: String
            let output: String
            let width: Int
            let height: Int
            let steps: Int
            let seed: Int
            let warmup: Bool
        }
        guard verifiedFile == nil, !published else {
            throw NativeFailure(message: "image_transaction_reused: 图片事务已使用。")
        }
        let summary: Summary
        do { summary = try JSONDecoder().decode(Summary.self, from: result) }
        catch { throw NativeFailure(message: "image_result_invalid: 生成器未返回有效图片结果。") }
        guard summary.schema_version == 1, !summary.warmup,
              summary.model == request.model, summary.operation == operation,
              summary.output == stagedURL.path,
              summary.width == request.width, summary.height == request.height,
              summary.steps == request.steps, summary.seed == request.seed else {
            throw NativeFailure(message: "image_result_mismatch: 图片结果与本次请求不一致。")
        }
        try Task.checkCancellation()
        let before = try identity()
        let fd = open(stagedURL.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else { throw NativeFailure(message: "image_artifact_invalid: 无法打开生成的图片。") }
        let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        guard try identity(fd: fd) == before else {
            throw NativeFailure(message: "image_artifact_changed: 图片在验证期间发生变化。")
        }
        try validatePNG(file, bytes: UInt64(before[2]))
        guard let source = CGImageSourceCreateWithURL(stagedURL as CFURL, nil),
              CGImageSourceGetType(source) as String? == "public.png",
              CGImageSourceGetCount(source) == 1,
              CGImageSourceGetStatus(source) == .statusComplete,
              let image = CGImageSourceCreateImageAtIndex(source, 0,
                  [kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
              image.width == request.width, image.height == request.height,
              let context = CGContext(data: nil, width: image.width, height: image.height,
                  bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw NativeFailure(message: "image_artifact_invalid: 图片格式、尺寸或解码结果无效。")
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete else {
            throw NativeFailure(message: "image_artifact_invalid: 图片数据不完整。")
        }
        var hash = SHA256()
        var remaining = UInt64(before[2])
        while remaining > 0 {
            try Task.checkCancellation()
            guard let chunk = try file.read(upToCount: Int(min(remaining, 1024 * 1024))), !chunk.isEmpty else {
                throw NativeFailure(message: "image_artifact_changed: 图片在验证期间被截断。")
            }
            hash.update(data: chunk)
            remaining -= UInt64(chunk.count)
        }
        guard try identity() == before, try identity(fd: fd) == before else {
            throw NativeFailure(message: "image_artifact_changed: 图片在验证期间发生变化。")
        }
        var receipt = try JSONSerialization.jsonObject(with: result) as! [String: Any]
        receipt["output"] = destination.path
        receipt["image_artifact"] = ["schema_version": 1, "staged_output": stagedURL.path,
            "published_output": destination.path, "bytes": before[2],
            "sha256": hash.finalize().map { String(format: "%02x", $0) }.joined(),
            "media_verified": true] as [String: Any]
        let data = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        verifiedIdentity = before
        verifiedFile = file
        return data
    }

    func publish() throws {
        try Task.checkCancellation()
        guard !published, let verifiedIdentity, let verifiedFile,
              try identity() == verifiedIdentity,
              try identity(fd: verifiedFile.fileDescriptor) == verifiedIdentity else {
            throw NativeFailure(message: "image_artifact_changed: 图片未经验证或已发生变化。")
        }
        guard rename(stagedURL.path, destination.path) == 0 else {
            throw NativeFailure(message: "image_publish_failed: 无法发布已验证的图片。")
        }
        published = true
    }
}
