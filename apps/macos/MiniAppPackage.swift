import Foundation
import CryptoKit
import zlib

struct MiniAppManifest: Codable, Equatable, Sendable {
    enum Permission: String, Codable, CaseIterable, Sendable {
        case importMedia = "media.import"
        case exportMedia = "media.export"
        case generate = "image.generate"
        case adapters = "adapters.use"

        var title: String {
            switch self {
            case .importMedia: "读取此应用中选择的图片"
            case .exportMedia: "保存此应用的图片和素材包"
            case .generate: "使用本机模型提交图片任务"
            case .adapters: "使用模型库中已登记的兼容 LoRA"
            }
        }
    }
    let schemaVersion: Int
    let sdkVersion: Int
    let id: String
    let name: String
    let version: String
    let entry: String
    let permissions: [Permission]
    let description: String?
}

struct MiniAppPackage: Identifiable, Sendable {
    static let sdkVersion = 1
    static let officialID = "org.turbocider.people-studio"
    static let maxExpandedBytes = 64 * 1024 * 1024
    private static let maxFileCount = 1024
    static let mimeTypes: [String: String] = [
        "html": "text/html", "js": "text/javascript", "mjs": "text/javascript", "css": "text/css",
        "json": "application/json", "txt": "text/plain", "md": "text/plain",
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
        "gif": "image/gif", "avif": "image/avif", "woff2": "font/woff2", "woff": "font/woff"
    ]
    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "gif", "avif"]
    let manifest: MiniAppManifest
    let files: [String: Data]
    let digest: String
    let builtIn: Bool
    let archiveData: Data?
    var id: String { manifest.id }

    init(source: URL, builtIn: Bool = false) throws {
        let access = source.startAccessingSecurityScopedResource()
        defer { if access { source.stopAccessingSecurityScopedResource() } }
        let meta = try source.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard meta.isSymbolicLink != true else { throw Self.invalid("应用包入口不能是符号链接。") }
        let entries: [String: Data]
        let archiveData: Data?
        if meta.isDirectory == true, builtIn {
            entries = try Self.readDirectory(source)
            archiveData = nil
        } else {
            guard meta.isRegularFile == true, source.pathExtension.lowercased() == "tcapp",
                  let size = meta.fileSize, (22...Self.maxExpandedBytes).contains(size) else {
                throw Self.invalid("请选择有效的 .tcapp 文件。")
            }
            let archive = try Data(contentsOf: source)
            entries = try Self.readZIP(archive)
            archiveData = archive
        }
        guard let manifestData = entries["manifest.json"], manifestData.count <= 16_384 else {
            throw Self.invalid("应用包缺少有效的 manifest.json。")
        }
        let manifest = try JSONDecoder().decode(MiniAppManifest.self, from: manifestData)
        let label = #"[a-z][a-z0-9-]*"#
        guard manifest.schemaVersion == 1, manifest.sdkVersion == Self.sdkVersion,
              manifest.id.utf8.count <= 96, manifest.id.range(of: "^\(label)(\\.\(label))+\\z", options: .regularExpression) != nil,
              manifest.version.utf8.count <= 32, manifest.version.range(of: #"^[0-9]+\.[0-9]+\.[0-9]+\z"#, options: .regularExpression) != nil,
              (1...64).contains(manifest.name.trimmingCharacters(in: .whitespacesAndNewlines).count),
              (manifest.description?.count ?? 0) <= 240,
              Set(manifest.permissions).count == manifest.permissions.count,
              Self.validPath(manifest.entry), (manifest.entry as NSString).pathExtension == "html", entries[manifest.entry] != nil,
              builtIn ? manifest.id == Self.officialID : !manifest.id.hasPrefix("org.turbocider.") && manifest.id != "org.turbocider" else {
            throw Self.invalid("应用清单的 ID、版本、权限或入口无效。")
        }
        var hasher = SHA256()
        for path in entries.keys.sorted() {
            let bytes = entries[path]!
            var count = UInt64(bytes.count).bigEndian
            hasher.update(data: Data(path.utf8))
            hasher.update(data: [UInt8(0)])
            withUnsafeBytes(of: &count) { hasher.update(bufferPointer: $0) }
            hasher.update(data: bytes)
        }
        self.digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        self.manifest = manifest
        self.files = entries
        self.builtIn = builtIn
        self.archiveData = archiveData
    }

    static func validPath(_ path: String) -> Bool {
        path.utf8.count <= 240 && !path.isEmpty && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { part in
            !part.isEmpty && !part.hasPrefix(".") && part.utf8.allSatisfy {
                (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0)
            }
        }
    }

    static func invalid(_ message: String) -> CocoaError {
        CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private static func validFile(_ path: String) -> Bool {
        validPath(path) && !path.hasPrefix("_media/") && !path.hasPrefix("_materials/") && mimeTypes[(path as NSString).pathExtension.lowercased()] != nil
    }

    private static func readDirectory(_ root: URL) throws -> [String: Data] {
        let keys: [URLResourceKey] = [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey]
        guard let items = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys) else {
            throw invalid("无法读取内置应用包。")
        }
        var result: [String: Data] = [:]
        var total = 0
        var paths = Set<String>()
        for case let file as URL in items {
            let key = String(file.path.dropFirst(root.path.count + 1))
            let values = try file.resourceValues(forKeys: Set(keys))
            guard values.isSymbolicLink != true, validPath(key) else { throw invalid("内置应用包包含无效路径。") }
            if values.isDirectory == true { continue }
            guard values.isRegularFile == true, let size = values.fileSize,
                  validFile(key), result.count < maxFileCount, size <= maxExpandedBytes - total,
                  paths.insert(key.lowercased()).inserted else { throw invalid("内置应用包包含无效文件。") }
            let data = try Data(contentsOf: file)
            guard data.count == size else { throw invalid("内置应用文件在读取期间发生变化。") }
            total += size
            result[key] = data
        }
        return result
    }

    private static func readZIP(_ data: Data) throws -> [String: Data] {
        func u16(_ p: Int) -> Int { Int(data[p]) | Int(data[p + 1]) << 8 }
        func u32(_ p: Int) -> Int { u16(p) | u16(p + 2) << 16 }
        guard (22...maxExpandedBytes).contains(data.count) else { throw invalid("应用包大小超限。") }
        let eocd = stride(from: data.count - 22, through: max(0, data.count - 22 - 65_535), by: -1)
            .first { u32($0) == 0x06054b50 && $0 + 22 + u16($0 + 20) == data.count }
        guard let end = eocd, u16(end + 4) == 0, u16(end + 6) == 0 else { throw invalid("应用包不是单卷 ZIP。") }
        let count = u16(end + 10), centralStart = u32(end + 16)
        guard count == u16(end + 8), (1...maxFileCount).contains(count),
              centralStart + u32(end + 12) == end else { throw invalid("应用包 ZIP 目录无效。") }
        var cursor = centralStart
        var result: [String: Data] = [:]
        var paths = Set<String>()
        var ranges: [Range<Int>] = []
        var total = 0
        for _ in 0..<count {
            guard cursor + 46 <= end, u32(cursor) == 0x02014b50 else { throw invalid("应用包的 ZIP 条目不完整。") }
            let flags = u16(cursor + 8), method = u16(cursor + 10), expectedCRC = u32(cursor + 16)
            let compressed = u32(cursor + 20), expanded = u32(cursor + 24)
            let nameSize = u16(cursor + 28), extra = u16(cursor + 30), comment = u16(cursor + 32)
            let disk = u16(cursor + 34), mode = (u32(cursor + 38) >> 16) & 0xf000
            let local = u32(cursor + 42)
            let nameStart = cursor + 46
            cursor = nameStart + nameSize + extra + comment
            guard cursor <= end, flags & ~0x80e == 0, [0, 8].contains(method), disk == 0,
                  [0, 0x4000, 0x8000].contains(mode),
                  let name = String(data: data[nameStart..<nameStart + nameSize], encoding: .utf8) else {
                throw invalid("应用包包含加密、特殊类型或不支持的 ZIP 文件。")
            }
            let directory = name.hasSuffix("/")
            let key = directory ? String(name.dropLast()) : name
            guard validPath(key), paths.insert(key.lowercased()).inserted,
                  local + 30 <= centralStart, u32(local) == 0x04034b50,
                  u16(local + 6) == flags, u16(local + 8) == method, u16(local + 26) == nameSize else {
                throw invalid("应用包包含重复路径或不一致的 ZIP 目录。")
            }
            let payload = local + 30 + nameSize + u16(local + 28)
            guard payload + compressed <= centralStart,
                  data[(local + 30)..<(local + 30 + nameSize)].elementsEqual(name.utf8),
                  expanded <= maxExpandedBytes - total else { throw invalid("应用包已损坏或解压后超限。") }
            ranges.append(local..<(payload + compressed))
            if directory {
                guard expanded == 0, mode != 0x8000 else { throw invalid("应用包目录项无效。") }
                continue
            }
            guard mode != 0x4000, validFile(key) else { throw invalid("应用包包含不支持的文件：\(key)") }
            let bytes = data.subdata(in: payload..<(payload + compressed))
            let file = try method == 8 ? inflate(bytes, expectedSize: expanded) : bytes
            let checksum = file.withUnsafeBytes {
                Int(crc32(0, $0.bindMemory(to: UInt8.self).baseAddress, uInt(file.count)))
            }
            guard file.count == expanded, checksum == expectedCRC else { throw invalid("应用包文件校验失败。") }
            result[key] = file
            total += expanded
        }
        let ordered = ranges.sorted { $0.lowerBound < $1.lowerBound }
        guard cursor == end, zip(ordered, ordered.dropFirst()).allSatisfy({ $0.upperBound <= $1.lowerBound }),
              !result.keys.contains(where: { path in
                  var parent = (path as NSString).deletingLastPathComponent
                  while !parent.isEmpty {
                      if result[parent] != nil { return true }
                      parent = (parent as NSString).deletingLastPathComponent
                  }
                  return false
              }) else { throw invalid("应用包 ZIP 布局无效。") }
        return result
    }

    private static func inflate(_ input: Data, expectedSize: Int) throws -> Data {
        var stream = z_stream()
        guard inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
            throw invalid("应用包解压初始化失败。")
        }
        defer { inflateEnd(&stream) }
        let capacity = expectedSize + 1
        var output = Data(count: capacity)
        let status = input.withUnsafeBytes { source in
            output.withUnsafeMutableBytes { destination in
                stream.next_in = UnsafeMutablePointer(mutating: source.bindMemory(to: UInt8.self).baseAddress)
                stream.avail_in = uInt(input.count)
                stream.next_out = destination.bindMemory(to: UInt8.self).baseAddress
                stream.avail_out = uInt(capacity)
                return zlib.inflate(&stream, Z_FINISH)
            }
        }
        guard status == Z_STREAM_END, stream.total_out == expectedSize, stream.total_in == input.count else {
            throw invalid("应用包解压大小或数据无效。")
        }
        output.removeLast()
        return output
    }
}
