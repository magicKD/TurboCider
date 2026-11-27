import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum PeopleMaterialCollection: String, CaseIterable, Codable, Sendable {
    case pose, depth, costume, anime, portrait, expr, hair, dyn, scene, light, prop, furn

    var title: String {
        switch self {
        case .pose: "姿势线稿"
        case .depth: "深度姿势"
        case .costume: "服装"
        case .anime: "动画服装"
        case .portrait: "人像"
        case .expr: "表情"
        case .hair: "发型"
        case .dyn: "历代服饰"
        case .scene: "场景"
        case .light: "光影"
        case .prop: "道具"
        case .furn: "家具"
        }
    }

    var file: String {
        switch self {
        case .pose: "poses"
        case .depth: "depths"
        case .costume: "costumes"
        case .anime: "anime"
        case .portrait: "portraits"
        case .expr: "expressions"
        case .hair: "hair"
        case .dyn: "dynasties"
        case .scene: "scenes"
        case .light: "lights"
        case .prop: "props"
        case .furn: "furniture"
        }
    }

    var downloadable: Bool { [.dyn, .scene, .light, .prop, .furn].contains(self) }
    var sourceID: String {
        switch self {
        case .pose: "original-pose"
        case .depth: "original-depth"
        case .costume: "original-costume"
        case .anime: "original-anime"
        case .portrait: "portrait"
        case .hair: "original-hair"
        case .expr: "posewear"
        case .dyn, .scene, .light, .prop, .furn: "shisei"
        }
    }
    var credit: String {
        switch self {
        case .pose, .depth, .costume, .anime: "ギア / @giaaibusi"
        case .portrait: "AI Portrait Gallery · @agi_aibusi"
        case .hair: "CharaDB · たき AI"
        case .expr: "WaveSpeed AI · ICT FaceKit"
        case .dyn, .scene, .light, .prop, .furn: "Shisei / wquguru · CC0"
        }
    }
}

enum PeopleMaterialSources {
    static let posewear = URL(string: "https://posewear.pages.dev/")!
    static let cdn = URL(string: "https://pwimg.0vo.pro/")!
    static let charaDB = URL(string: "https://charadb-2fi.pages.dev/")!
    private static let collections = Set(["p", "r", "c", "a", "e", "y", "s", "l", "t", "f"])

    static let pages: [String: URL] = [
        "shisei": URL(string: "https://shisei.indaloop.app/")!,
        "portrait": URL(string: "https://ai-portrait-gallery.pages.dev/")!,
        "posewear": posewear,
        "lanpose": URL(string: "https://github.com/wzc520pyfm/gen-ai-pose-image")!,
        "original-pose": URL(string: "https://ai-pose-gallery.pages.dev/")!,
        "original-depth": URL(string: "https://ai-pose-depth-gallery.pages.dev/")!,
        "original-costume": URL(string: "https://ai-costume-gallery.pages.dev/")!,
        "original-anime": URL(string: "https://ai-costume-anime-gallery.pages.dev/")!,
        "original-hair": charaDB
    ]

    static func allows(_ url: URL) -> Bool {
        guard url.scheme == "https", url.port == nil, url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil else { return false }
        if url == posewear { return true }
        let path = String(url.path.dropFirst())
        guard MiniAppPackage.validPath(path) else { return false }
        let parts = path.split(separator: "/").map(String.init)
        if url.host == posewear.host {
            if parts.count == 2, parts[0] == "assets" {
                return parts[1].range(of: #"^(app\.[a-f0-9]{10,64}\.js|[a-z]+\.[a-f0-9]{10,64}\.json)\z"#,
                                      options: .regularExpression) != nil
            }
            return parts.count == 3 && collections.contains(parts[0]) && ["m", "i"].contains(parts[1]) && image(url)
        }
        if url.host == cdn.host {
            guard parts.count == 3, image(url) else { return false }
            return (collections.contains(parts[0]) && parts[1] == "t") ||
                (parts[0] == "o" && collections.contains(parts[1])) ||
                (parts[0] == "d" && ["thumbs", "img"].contains(parts[1]))
        }
        return url.host == charaDB.host && parts.count == 2 && ["thumb", "img"].contains(parts[0]) && image(url)
    }

    static func image(_ path: String?) -> URL? {
        guard let path, !path.isEmpty else { return nil }
        let url: URL
        if let absolute = URL(string: path), absolute.scheme != nil { url = absolute }
        else {
            let relative = path.hasPrefix("/") ? String(path.dropFirst()) : path
            guard MiniAppPackage.validPath(relative) else { return nil }
            let parts = relative.split(separator: "/")
            guard parts.count == 3 else { return nil }
            let host = parts[0] == "o" || parts[1] == "t" ? cdn : posewear
            url = host.appendingPathComponent(relative)
        }
        return image(url) && allows(url) ? url : nil
    }

    private static func image(_ url: URL) -> Bool {
        ["webp", "png", "jpg", "jpeg"].contains(url.pathExtension.lowercased())
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private final class PeopleMaterialNoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

private enum PeopleMaterialHTTP {
    static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.httpMaximumConnectionsPerHost = 4
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 15
        configuration.timeoutIntervalForResource = 30
        return URLSession(configuration: configuration, delegate: PeopleMaterialNoRedirect(), delegateQueue: nil)
    }()

    static func read(_ url: URL, limit: Int) async throws -> Data {
        guard PeopleMaterialSources.allows(url) else { throw PeopleStudioMaterials.Failure.invalidSource }
        var request = URLRequest(url: url)
        request.setValue("TurboCider-Materials/1.0", forHTTPHeaderField: "User-Agent")
        let (stream, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, http.url == url else {
            throw PeopleStudioMaterials.Failure.invalidSource
        }
        guard http.statusCode == 200 else { throw PeopleStudioMaterials.Failure.http(http.statusCode) }
        guard response.expectedContentLength <= Int64(limit) else { throw PeopleStudioMaterials.Failure.tooLarge }
        var data = Data()
        if response.expectedContentLength > 0 { data.reserveCapacity(Int(response.expectedContentLength)) }
        for try await byte in stream {
            guard data.count < limit else { throw PeopleStudioMaterials.Failure.tooLarge }
            data.append(byte)
        }
        guard !data.isEmpty else { throw PeopleStudioMaterials.Failure.invalidImage }
        return data
    }
}

actor PeopleStudioMaterials {
    enum Failure: LocalizedError {
        case invalidSource, invalidCatalog, invalidImage, disabled, notAvailable, notDownloadable, previewOnly, tooLarge, systemDownloadBusy, http(Int)
        var errorDescription: String? {
            switch self {
            case .invalidSource: "素材地址不在已接入的来源中。"
            case .invalidCatalog: "上游目录格式已变化，暂时无法载入。"
            case .invalidImage: "上游文件无法解码为静态参考图。"
            case .disabled: "请先开启在线素材，或选择已下载的素材。"
            case .notAvailable: "素材已失效，请刷新目录后重新选择。"
            case .notDownloadable: "这套素材仅支持从来源按需访问。"
            case .previewOnly: "这张 CharaDB 图片仅支持原站预览。请选择其他带参考图的素材，或导入你有权使用的图片。"
            case .tooLarge: "素材或本地包超出大小限制。"
            case .systemDownloadBusy: "请等当前素材库下载结束，或先停止下载。"
            case .http(let status): "素材来源暂时不可访问（HTTP \(status)）。"
            }
        }
    }

    struct Picture: Codable, Sendable {
        let thumb: URL
        let full: URL
        let previewOnly: Bool
    }

    struct Catalog: Codable, Sendable {
        var version = 1
        let revision: String
        let entries: [Data]
        let pictures: [String: Picture]
    }

    private struct CatalogFile: Sendable {
        let url: URL
        let fingerprint: String
    }

    struct Resource: Sendable {
        let data: Data
        let mime: String
        let name: String
    }

    struct Status: Encodable, Sendable {
        let online: Bool
        let collections: [CollectionStatus]
    }

    struct CollectionStatus: Encodable, Sendable {
        let id: PeopleMaterialCollection
        let title: String
        let downloadable: Bool
        let count: Int?
        let phase: String
        let completed: Int
        let total: Int
        let bytes: Int
        let error: String?
    }

    private struct Download {
        var phase = "downloading"
        var completed = 0
        var total = 0
        var bytes = 0
        var error: String?
    }

    typealias Reader = @Sendable (URL, Int) async throws -> Data
    private let root: URL
    private let read: Reader
    private var online = false
    private var loaded: [PeopleMaterialCollection: Catalog] = [:]
    private var installed: [PeopleMaterialCollection: Catalog] = [:]
    private var checkedInstall = Set<PeopleMaterialCollection>()
    private var catalogIndex: [PeopleMaterialCollection: CatalogFile]?
    private var indexTask: Task<[PeopleMaterialCollection: CatalogFile], Error>?
    private var fetching: [PeopleMaterialCollection: Task<Catalog, Error>] = [:]
    private var downloading: (collection: PeopleMaterialCollection, task: Task<Void, Never>)?
    private var progress: [PeopleMaterialCollection: Download] = [:]
    private let previews = NSCache<NSString, NSData>()
    private static let imageLimit = 16 * 1024 * 1024
    private static let packLimit = 128 * 1024 * 1024

    init(root: URL, reader: Reader? = nil) {
        self.root = root
        self.read = reader ?? { try await PeopleMaterialHTTP.read($0, limit: $1) }
        previews.totalCostLimit = 24 * 1024 * 1024
    }

    func enableOnline(_ enabled: Bool) {
        online = enabled
        if !enabled {
            previews.removeAllObjects()
            cancelCatalogReads(keeping: downloading?.collection)
        }
    }

    func refresh() {
        guard downloading == nil else { return }
        cancelCatalogReads()
        loaded.removeAll()
        catalogIndex = nil
        previews.removeAllObjects()
    }

    private func cancelCatalogReads(keeping collection: PeopleMaterialCollection? = nil) {
        if collection == nil {
            indexTask?.cancel()
            indexTask = nil
        }
        for id in fetching.keys where id != collection {
            fetching.removeValue(forKey: id)?.cancel()
        }
    }

    func status() -> Status {
        Status(online: online, collections: PeopleMaterialCollection.allCases.map { collection in
            let complete = localCatalog(collection)
            let state = progress[collection] ?? Download(phase: complete == nil ? "idle" : "installed")
            return CollectionStatus(id: collection, title: collection.title, downloadable: collection.downloadable,
                count: (complete ?? loaded[collection])?.entries.count, phase: state.phase,
                completed: state.completed, total: state.total, bytes: state.bytes, error: state.error)
        })
    }

    func rows(_ collection: PeopleMaterialCollection) async throws -> [Data] {
        if let offline = localCatalog(collection) { return offline.entries }
        guard online else { return [] }
        let remote = try await remoteCatalog(collection)
        return online ? remote.entries : []
    }

    func sourceImage(_ token: String, full: Bool, forReference: Bool = false) async throws -> Resource {
        guard let (collection, key) = Self.parse(token) else { throw Failure.invalidSource }
        if let catalog = localCatalog(collection), let picture = catalog.pictures[key],
           let data = try? Self.readFile(file(key, in: packDirectory(collection)), limit: Self.imageLimit),
           let result = try? Self.validImage(data, url: picture.full) {
            return result
        }
        guard online else { throw Failure.disabled }
        guard let picture = loaded[collection]?.pictures[key] else { throw Failure.notAvailable }
        if forReference && picture.previewOnly { throw Failure.previewOnly }
        let url = full ? picture.full : picture.thumb
        guard PeopleMaterialSources.allows(url) else { throw Failure.invalidSource }
        let cacheKey = url.absoluteString as NSString
        if let data = previews.object(forKey: cacheKey) as Data? { return try Self.validImage(data, url: url) }
        let data = try await read(url, Self.imageLimit)
        try Task.checkCancellation()
        guard online else { throw Failure.disabled }
        let result = try Self.validImage(data, url: url)
        previews.setObject(data as NSData, forKey: cacheKey, cost: data.count)
        return result
    }

    func startDownload(_ collection: PeopleMaterialCollection) throws {
        guard collection.downloadable else { throw Failure.notDownloadable }
        guard downloading == nil else { throw Failure.systemDownloadBusy }
        guard localCatalog(collection) == nil else { return }
        progress[collection] = Download()
        let task = Task { [weak self] in await self?.download(collection) ?? () }
        downloading = (collection, task)
    }

    func cancelDownload(_ collection: PeopleMaterialCollection) {
        guard downloading?.collection == collection else { return }
        downloading?.task.cancel()
        if !online { cancelCatalogReads() }
    }

    private func remoteIndex() async throws -> [PeopleMaterialCollection: CatalogFile] {
        if let catalogIndex { return catalogIndex }
        if let pending = indexTask { return try await pending.value }
        let reader = read
        let task = Task {
            defer { if !Task.isCancelled { indexTask = nil } }
            let home = try await reader(PeopleMaterialSources.posewear, 512 * 1024)
            try Task.checkCancellation()
            let html = String(decoding: home, as: UTF8.self)
            guard let app = Self.matches(#"src=["'](/assets/app\.([a-f0-9]{10,64})\.js)["']"#, in: html).first,
                  let url = URL(string: app[0], relativeTo: PeopleMaterialSources.posewear)?.absoluteURL else {
                throw Failure.invalidCatalog
            }
            let bytes = try await reader(url, 1024 * 1024)
            guard PeopleMaterialSources.sha256(bytes).hasPrefix(app[1]) else { throw Failure.invalidCatalog }
            let script = String(decoding: bytes, as: UTF8.self)
            let candidates = Self.matches(#"["'](/assets/([a-z]+)\.([a-f0-9]{10,64})\.json)["']"#, in: script)
            var result: [PeopleMaterialCollection: CatalogFile] = [:]
            for (path, name, hash) in candidates.map({ ($0[0], $0[1], $0[2]) }) {
                if let collection = PeopleMaterialCollection.allCases.first(where: { $0.file == name }),
                   let url = URL(string: path, relativeTo: PeopleMaterialSources.posewear)?.absoluteURL {
                    result[collection] = CatalogFile(url: url, fingerprint: hash)
                }
            }
            guard !result.isEmpty else { throw Failure.invalidCatalog }
            try Task.checkCancellation()
            catalogIndex = result
            return result
        }
        indexTask = task
        return try await task.value
    }

    private func remoteCatalog(_ collection: PeopleMaterialCollection) async throws -> Catalog {
        if let previous = loaded[collection] { return previous }
        if let pending = fetching[collection] { return try await pending.value }
        let reader = read
        let task = Task {
            defer { if !Task.isCancelled { fetching[collection] = nil } }
            let index = try await remoteIndex()
            try Task.checkCancellation()
            guard let file = index[collection] else { throw Failure.invalidCatalog }
            let data = try await reader(file.url, 3 * 1024 * 1024)
            try Task.checkCancellation()
            guard PeopleMaterialSources.sha256(data).hasPrefix(file.fingerprint) else { throw Failure.invalidCatalog }
            let result = try Self.catalog(collection, data: data, revision: file.fingerprint)
            loaded[collection] = result
            return result
        }
        fetching[collection] = task
        return try await task.value
    }

    private func packDirectory(_ collection: PeopleMaterialCollection, revision: String? = nil) -> URL {
        let prefix = "posewear-v1-" + collection.rawValue
        return root.appendingPathComponent(prefix + (revision.map { "-\($0).partial" } ?? ""), isDirectory: true)
    }

    private func file(_ key: String, in directory: URL) -> URL {
        directory.appendingPathComponent(key.replacingOccurrences(of: "/", with: "--") + ".image")
    }

    private func localCatalog(_ collection: PeopleMaterialCollection) -> Catalog? {
        if let catalog = installed[collection] { return catalog }
        guard collection.downloadable, checkedInstall.insert(collection).inserted,
              let bytes = try? Self.readFile(packDirectory(collection).appendingPathComponent("index.json"), limit: 8 * 1024 * 1024),
              let value = try? JSONDecoder().decode(Catalog.self, from: bytes),
              value.version == 1 else { return nil }
        installed[collection] = value
        return value
    }

    private func download(_ collection: PeopleMaterialCollection) async {
        defer { downloading = nil }
        do {
            let catalog = try await remoteCatalog(collection)
            try Task.checkCancellation()
            let partial = packDirectory(collection, revision: catalog.revision)
            try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
            let images = catalog.pictures.sorted { $0.key < $1.key }
            progress[collection]?.total = images.count
            var count = 0
            var size = 0
            let reader = read
            // Limit concurrent downloads to the same budget used by the live gallery.
            for offset in stride(from: 0, to: images.count, by: 4) {
                try Task.checkCancellation()
                let end = min(offset + 4, images.count)
                try await withThrowingTaskGroup(of: Int.self) { group in
                    for (key, picture) in images[offset..<end] {
                        let destination = file(key, in: partial)
                        group.addTask {
                            if let existing = try? Self.readFile(destination, limit: Self.imageLimit),
                               (try? Self.validImage(existing, url: picture.full)) != nil {
                                return existing.count
                            }
                            guard !picture.previewOnly, PeopleMaterialSources.allows(picture.full) else { throw Failure.invalidSource }
                            let data = try await reader(picture.full, Self.imageLimit)
                            _ = try Self.validImage(data, url: picture.full)
                            try Task.checkCancellation()
                            try data.write(to: destination, options: .atomic)
                            return data.count
                        }
                    }
                    for try await bytes in group {
                        size += bytes
                        guard size <= Self.packLimit else { throw Failure.tooLarge }
                        count += 1
                        progress[collection]?.completed = count
                        progress[collection]?.bytes = size
                    }
                }
            }
            try Task.checkCancellation()
            let index = try JSONEncoder().encode(catalog)
            try index.write(to: partial.appendingPathComponent("index.json"), options: .atomic)
            try FileManager.default.moveItem(at: partial, to: packDirectory(collection))
            installed[collection] = catalog
            progress[collection]?.phase = "installed"
        } catch is CancellationError {
            progress[collection]?.phase = "cancelled"
        } catch {
            if Task.isCancelled { progress[collection]?.phase = "cancelled" }
            else {
                progress[collection]?.phase = "failed"
                progress[collection]?.error = error.localizedDescription
            }
        }
    }

    private static func parse(_ token: String) -> (PeopleMaterialCollection, String)? {
        let fields = token.split(separator: "/", omittingEmptySubsequences: false)
        guard fields.count == 3, MiniAppPackage.validPath(token),
              let collection = PeopleMaterialCollection(rawValue: String(fields[0])) else { return nil }
        return (collection, token)
    }

    private static func readFile(_ url: URL, limit: Int) throws -> Data {
        let metadata = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard metadata.isRegularFile == true, metadata.isSymbolicLink != true,
              let size = metadata.fileSize, (1...limit).contains(size) else { throw Failure.invalidImage }
        return try Data(contentsOf: url, options: .mappedIfSafe)
    }

    private static func validImage(_ data: Data, url: URL) throws -> Resource {
        guard (1...imageLimit).contains(data.count),
              let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) == 1,
              let identifier = CGImageSourceGetType(source) as String?,
              let mime = UTType(identifier)?.preferredMIMEType, ["image/webp", "image/png", "image/jpeg"].contains(mime),
              let info = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = info[kCGImagePropertyPixelWidth] as? Int, let height = info[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, Double(width) * Double(height) <= 80_000_000 else { throw Failure.invalidImage }
        return Resource(data: data, mime: mime, name: url.lastPathComponent)
    }
}

extension PeopleStudioMaterials {
    private static func matches(_ expression: String, in text: String) -> [[String]] {
        guard let pattern = try? NSRegularExpression(pattern: expression) else { return [] }
        return pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)).map { match in
            (1..<match.numberOfRanges).map { index in
                Range(match.range(at: index), in: text).map { String(text[$0]) } ?? ""
            }
        }
    }

    static func catalog(_ collection: PeopleMaterialCollection, data: Data, revision: String) throws -> Catalog {
        let raw = try JSONSerialization.jsonObject(with: data)
        guard let rows = (raw as? [[String: Any]]) ?? (raw as? [String: Any])?["items"] as? [[String: Any]],
              rows.count <= 2_500 else { throw Failure.invalidCatalog }
        var entries: [Data] = []
        var pictures: [String: Picture] = [:]
        var ids = Set<String>()
        let textFields = ["no", "ti", "zh", "en", "ja", "kw", "line", "lineEn", "lineJa", "cat", "catEn", "catJa", "era",
            "eraEn", "gender", "emo", "emoEn", "int", "intEn", "len", "sec", "st", "setting", "light", "dims"]
        let angleFields = ["zh", "en", "ja"]

        func limitedText(_ source: [String: Any], keys: [String]) -> [String: Any] {
            var output: [String: Any] = [:]
            for key in keys {
                if let value = source[key] as? String { output[key] = String(value.prefix(1000)) }
            }
            return output
        }

        for row in rows {
            guard let id = row["id"] as? String, id.utf8.count <= 96, !id.contains("/"), MiniAppPackage.validPath(id) else { continue }
            guard ids.insert(id).inserted else { throw Failure.invalidCatalog }
            var result = limitedText(row, keys: textFields)
            result["id"] = id
            for key in ["w", "h", "rk", "rc"] {
                if let value = row[key] as? Int, (0...100_000).contains(value) { result[key] = value }
            }
            if row["a"] as? Bool == true { result["a"] = true }
            if collection == .depth && row["pack"] as? String == "gia-depth" { result["pack"] = "gia-depth" }
            let variants = row["angles"] as? [[String: Any]]
            if (variants?.count ?? 0) > 24 { continue }
            var images: [[String: Any]] = []
            var directPreview = false
            for (position, variant) in (variants ?? [row]).enumerated() {
                guard let thumb = PeopleMaterialSources.image(variant["t"] as? String),
                      let full = PeopleMaterialSources.image(variant["i"] as? String ?? variant["m"] as? String ?? variant["o"] as? String) else { continue }
                let previewOnly = thumb.host == PeopleMaterialSources.charaDB.host || full.host == PeopleMaterialSources.charaDB.host
                if collection.downloadable && previewOnly { continue }
                directPreview = directPreview || previewOnly
                let token = "\(collection.rawValue)/\(id)/v\(position)"
                pictures[token] = Picture(thumb: thumb, full: full, previewOnly: previewOnly)
                let base = "/_materials/" + token
                var item = limitedText(variant, keys: angleFields)
                item["t"] = base + "/thumb"
                for key in ["i", "m", "o"] { item[key] = base + "/full" }
                item["materialID"] = token
                item["sourceURL"] = full.absoluteString
                images.append(item)
            }
            guard let first = images.first else { continue }
            for key in ["t", "i", "m", "o", "materialID", "sourceURL"] { result[key] = first[key] }
            if variants != nil { result["angles"] = images }
            result["materialSource"] = directPreview ? "original-hair" : collection.sourceID
            result["materialCredit"] = directPreview ? PeopleMaterialCollection.hair.credit : collection.credit
            result["materialDownloadable"] = collection.downloadable
            result["previewOnly"] = directPreview
            entries.append(try JSONSerialization.data(withJSONObject: result))
        }
        guard !entries.isEmpty else { throw Failure.invalidCatalog }
        return Catalog(revision: revision, entries: entries, pictures: pictures)
    }
}
