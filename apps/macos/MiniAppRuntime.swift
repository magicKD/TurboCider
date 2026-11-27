import AppKit
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct MiniAppReference: Codable, Sendable {
    enum Role: String, Codable, Sendable {
        case person, garment, pose, depth, expression, hairstyle, scene, lighting, prop, furniture, reference
        var instruction: String {
            switch self {
            case .person: "person identity, facial features and body proportions"
            case .garment: "clothing design, color, fabric and fit"
            case .pose: "pose and framing only"
            case .depth: "depth and body geometry only"
            case .expression: "facial expression"
            case .hairstyle: "hairstyle"
            case .scene: "background and scene"
            case .lighting: "lighting and shadows"
            case .prop: "prop design and placement"
            case .furniture: "furniture design and placement"
            case .reference: "visual details as requested"
            }
        }
    }
    let mediaID: UUID?
    let shotIndex: Int?
    let role: Role
}

struct MiniAppShot: Codable, Sendable {
    struct Adapter: Codable, Sendable {
        let id: String
        let strength: Double?
    }
    let prompt: String
    let references: [MiniAppReference]
    let width: Int?
    let height: Int?
    let matchReferenceAspect: Bool?
    let referencePixelBudget: Int?
    let seed: Int?
    let adapters: [Adapter]?
    let label: String?
    let context: String?

    func native(model: String, variant: String?, steps: Int, output: URL,
                assets: [StudioAsset], loras: [NativeLoRA], supportsWeightStreaming: Bool = false) throws -> NativeRequest {
        let qwen = model == Qwen21Installation.modelID
        let matched = matchReferenceAspect == true && !(qwen && !loras.isEmpty) ? matchedCanvas(assets) : nil
        let width = self.width ?? matched?.width ?? 512, height = self.height ?? matched?.height ?? 512
        let minimumDimension = matchReferenceAspect == true ? 64 : 256
        let maximumPixels = referencePixelBudget == nil ? 1_048_576 : 768 * 512
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, prompt.utf8.count <= 12_000, !prompt.contains("\0"),
              references.count == assets.count, references.count <= 10, (label?.count ?? 0) <= 160, (context?.utf8.count ?? 0) <= 8_192,
              matchReferenceAspect != true || (self.width == nil && self.height == nil),
              referencePixelBudget.map({ variant == Qwen21Installation.turboID && $0 >= max(1, assets.count) * 1024 && $0 <= 512 * 512 && $0 % 1024 == 0 }) ?? true,
              (minimumDimension...2048).contains(width), (minimumDimension...2048).contains(height), width % 32 == 0, height % 32 == 0,
              width * height <= maximumPixels,
              seed.map({ (0...Int(Int32.max)).contains($0) }) ?? true else {
            throw MiniAppRuntime.Failure("invalidRequest", "提示词、尺寸、参考图数量或随机种子超出应用任务支持范围。")
        }
        let instructions: [String]
        if qwen {
            instructions = references.enumerated().map { "Use <image\($0.offset + 1)> for \($0.element.role.instruction)." }
        } else {
            instructions = references.enumerated().map { "Use reference image \($0.offset + 1) for \($0.element.role.instruction)." }
        }
        let composed = instructions.isEmpty ? prompt : instructions.joined(separator: " ") + "\n\n" + prompt
        var request = NativeRequest(prompt: composed, output: output.path)
        request.model = model
        request.model_variant = variant
        request.operation = assets.isEmpty ? "image.generate" : "image.edit"
        request.inputs = assets.map { NativeInput(kind: "image", role: "reference", path: $0.path) }
        request.width = width; request.height = height; request.steps = steps
        request.seed = seed ?? Int.random(in: 0...Int(Int32.max))
        request.residency = referencePixelBudget != nil && supportsWeightStreaming && loras.isEmpty ? "streamed" : "component_staged"
        if qwen {
            request.allow_approximation = true
            request.qwen21_reference_pixel_budget = assets.isEmpty ? nil : referencePixelBudget
            request.qwen21_reference_size = referencePixelBudget == nil ? 1024 : nil
        }
        if !loras.isEmpty {
            request.loras = loras
            request.lora_strategy = "inference_time"
        }
        return request
    }

    private func matchedCanvas(_ assets: [StudioAsset]) -> EditingCanvasDimensions? {
        let index = references.firstIndex { $0.role == .person } ??
            references.firstIndex { $0.role == .pose || $0.role == .depth } ?? 0
        guard assets.indices.contains(index) else { return nil }
        let source = assets[index].originalImage
        guard source.width > 0, source.height > 0 else { return nil }
        let dimensions = EditingCanvasDimensions(width: source.width, height: source.height)
        let area = Double(source.width) * Double(source.height)
        let scale = min(2048 / Double(max(source.width, source.height)), sqrt(Double(512 * 512) / area))
        let fitted = EditingCanvasSizing.alignedDimensions(dimensions, scale: scale, multiple: 32, shrinkOnly: false)
        return EditingCanvasDimensions(width: max(64, fitted.width), height: max(64, fitted.height))
    }
}

@MainActor
final class MiniAppRuntime: ObservableObject {
    private struct Scope: Hashable, Sendable {
        let app: String
        let device: UUID?

        func owns(_ origin: MiniAppJobOrigin?) -> Bool { origin?.appID == app && origin?.deviceID == device }
        func mediaURL(_ id: UUID) -> String {
            if device != nil { return "/apps/\(app)/_media/\(id.uuidString)" }
            return "tcapp://\(app)/_media/\(id.uuidString)"
        }
    }
    struct Failure: LocalizedError {
        let code: String
        let message: String
        init(_ code: String, _ message: String) { self.code = code; self.message = message }
        var errorDescription: String? { message }
    }
    private struct MediaRecord: Codable {
        var asset: StudioAsset
        var bundledPath: String? = nil
        var bundledDigest: String? = nil
    }
    private struct PrepareRequest: Codable {
        let requestID: UUID
        let workflow: String
        let engine: String?
        let shots: [MiniAppShot]
    }
    private struct Plan: Sendable {
        let scope: Scope
        let appDigest: String
        let requestID: UUID
        let requestDigest: String
        let jobs: [NativeJob]
        let created: Date
    }
    private struct IDRequest: Decodable { let id: UUID }
    private struct DataRequest: Decodable { let data: String; let name: String? }
    private struct PathRequest: Decodable { let path: String }
    private struct StorageRequest: Decodable { let key: String; let value: String? }
    static let maxImageBytes = 16 * 1024 * 1024
    private static let maxExportBytes = 24 * 1024 * 1024

    @Published private(set) var packages: [MiniAppPackage] = []
    @Published var message: String?
    private let root: URL
    private let peopleMaterials: PeopleStudioMaterials
    private var appsFolder: URL { root.appendingPathComponent("packages", isDirectory: true) }
    let store: NativeJobStore
    let studio: StudioState
    private var media: [Scope: [UUID: MediaRecord]] = [:]
    private var importers: [Scope: StudioAssetImporter] = [:]
    private var plans: [UUID: Plan] = [:]
    private var planningApps = Set<Scope>()
    private var importingApps = Set<Scope>()
    private var panelActive = false
    private var configuredTurboModel: String?
    private var modelPreference: URL { root.appendingPathComponent("qwen-turbo.json") }

    init(store: NativeJobStore, studio: StudioState, builtIn: URL?, materials: PeopleStudioMaterials? = nil) {
        self.store = store
        self.studio = studio
        let miniAppRoot = store.directory.appendingPathComponent("mini-apps", isDirectory: true)
        self.root = miniAppRoot
        self.peopleMaterials = materials ?? PeopleStudioMaterials(root: miniAppRoot.appendingPathComponent("people-materials", isDirectory: true))
        if let builtIn {
            do { packages.append(try MiniAppPackage(source: builtIn, builtIn: true)) }
            catch { message = "无法载入人物工作室。\(error.localizedDescription)" }
        }
        if let files = try? FileManager.default.contentsOfDirectory(at: appsFolder, includingPropertiesForKeys: nil) {
            for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }).prefix(32) where file.pathExtension == "tcapp" {
                do {
                    let package = try MiniAppPackage(source: file)
                    guard file.deletingPathExtension().lastPathComponent == package.id else {
                        throw Failure("invalidPackage", "已安装应用 ID 与文件名不一致。")
                    }
                    packages.append(package)
                } catch { message = "有社区应用未能载入。\(error.localizedDescription)" }
            }
        }
        if let data = try? Data(contentsOf: modelPreference) {
            configuredTurboModel = try? JSONDecoder().decode(String.self, from: data)
        }
    }

    func package(_ id: String) -> MiniAppPackage? { packages.first { $0.id == id } }

    func install(_ package: MiniAppPackage) throws {
        guard !package.builtIn, let archive = package.archiveData,
              self.package(package.id) != nil || packages.filter({ !$0.builtIn }).count < 32 else {
            throw Failure("installLimit", "已达到社区应用安装数量上限。")
        }
        try FileManager.default.createDirectory(at: appsFolder, withIntermediateDirectories: true)
        try archive.write(to: appsFolder.appendingPathComponent(package.id + ".tcapp"), options: .atomic)
        packages.removeAll { $0.id == package.id }
        packages.append(package)
        plans = plans.filter { $0.value.scope.app != package.id }
    }

    func uninstall(_ package: MiniAppPackage) throws {
        guard !package.builtIn else { return }
        try FileManager.default.removeItem(at: appsFolder.appendingPathComponent(package.id + ".tcapp"))
        packages.removeAll { $0.id == package.id }
        plans = plans.filter { $0.value.scope.app != package.id }
    }

    private func library() throws -> LibraryIndex {
        try LibraryStore(createIfMissing: false).read()
    }

    private func installation(_ engine: String) throws -> (URL, model: String, variant: String?, steps: Int) {
        let model: String
        let variant: String?
        let steps: Int
        switch engine {
        case Qwen21Installation.turboID:
            model = Qwen21Installation.modelID; variant = Qwen21Installation.turboID; steps = 8
        case Qwen21Installation.modelID:
            model = engine; variant = engine; steps = 40
        case "flux2-klein-4b", "z-image-turbo":
            model = engine; variant = nil
            steps = studio.models.first(where: { $0.id == model })?.default_steps ?? 4
        default: throw Failure("unsupportedModel", "应用宿主不支持此图片模型。")
        }
        let registered = try library().installations.filter { $0.modelID == model }.map(\.path)
        let primary = variant == Qwen21Installation.turboID ? configuredTurboModel : nil
        let candidates = [primary, studio.draft.modelPaths[model]].compactMap { $0 } + registered
        let path = candidates.first { path in
            guard path.hasPrefix("/"), FileManager.default.fileExists(atPath: path) else { return false }
            if model == Qwen21Installation.modelID {
                guard Qwen21Installation.checkpointName(at: path) != nil else { return false }
                return Qwen21Installation.isTurbo(at: path) == (variant == Qwen21Installation.turboID)
            }
            return true
        }
        guard let path else {
            let hint = engine == Qwen21Installation.turboID ? "也可以在人物工作室选择本机 Turbo 模型目录。" : ""
            throw Failure("modelMissing", "请先在「模型」中安装 \(engine)。" + hint)
        }
        return (URL(fileURLWithPath: path, isDirectory: true).resolvingSymlinksInPath(), model, variant, steps)
    }

    func chooseTurboModel() async throws -> Bool {
        let panel = NSOpenPanel()
        panel.title = "选择 Qwen Image 2.1 Turbo 模型目录"
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.allowsMultipleSelection = false
        guard try await runPanel(panel), let url = panel.url else { return false }
        let path = url.resolvingSymlinksInPath().path
        guard Qwen21Installation.isTurbo(at: path) else {
            throw Failure("modelMissing", "这个目录没有可识别的 Qwen Image 2.1 Turbo 权重。")
        }
        var probe = NativeRequest(prompt: "A portrait in natural light.", output: store.directory.appendingPathComponent("outputs/mini-app-probe.png").path)
        probe.model = Qwen21Installation.modelID; probe.model_variant = Qwen21Installation.turboID
        probe.steps = 8; probe.allow_approximation = true; probe.residency = "component_staged"
        let request = probe
        _ = try await Task.detached { try NativeEngine.plan(request, installation: url) }.value
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try JSONEncoder().encode(path).write(to: modelPreference, options: .atomic)
        configuredTurboModel = path
        return true
    }

    private func dataDirectory(_ scope: Scope) -> URL {
        let path = scope.device.map { "browser/\($0.uuidString)/\(scope.app)" } ?? "data/\(scope.app)"
        return root.appendingPathComponent(path, isDirectory: true)
    }
    private func records(_ scope: Scope) throws -> [UUID: MediaRecord] {
        if let cached = media[scope] { return cached }
        let file = dataDirectory(scope).appendingPathComponent("media.json")
        let value = FileManager.default.fileExists(atPath: file.path)
            ? try JSONDecoder().decode([UUID: MediaRecord].self, from: Data(contentsOf: file)) : [:]
        media[scope] = value
        return value
    }

    private func descriptor(_ asset: StudioAsset, id: UUID, scope: Scope) -> [String: Any] {
        ["id": id.uuidString, "url": scope.mediaURL(id),
         "name": asset.name, "width": asset.width, "height": asset.height]
    }

    private func resultAsset(id: UUID, scope: Scope) -> StudioAsset? {
        guard let job = store.jobs.first(where: { $0.id == id && scope.owns($0.miniAppOrigin) }), job.hasOutput else { return nil }
        let output = URL(fileURLWithPath: job.request.output)
        return StudioAsset(path: output.path, name: output.lastPathComponent, width: job.request.width, height: job.request.height)
    }

    private func asset(id: UUID, scope: Scope) throws -> StudioAsset {
        guard let record = try records(scope)[id]?.asset ?? resultAsset(id: id, scope: scope),
              FileManager.default.isReadableFile(atPath: record.path) else {
            throw Failure("mediaNotFound", "找不到此应用的图片。请重新导入。")
        }
        return record
    }

    func mediaFile(_ id: UUID, package: MiniAppPackage, device: UUID? = nil, exporting: Bool = false) throws -> (URL, String) {
        try requireCurrent(package)
        if exporting { try authorize(.exportMedia, package) }
        let asset = try asset(id: id, scope: Scope(app: package.id, device: device))
        let path = URL(fileURLWithPath: asset.path).resolvingSymlinksInPath()
        let type = MiniAppPackage.mimeTypes[path.pathExtension.lowercased()]
            ?? UTType(filenameExtension: path.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
        let values = try path.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard type.hasPrefix("image/"), values.isRegularFile == true,
              let size = values.fileSize, size > 0, size <= 32 * 1024 * 1024 else {
            throw Failure("mediaNotFound", "图片格式或大小超出应用读取范围。")
        }
        return (path, type)
    }

    func materialResource(_ path: String, app: MiniAppPackage) async throws -> PeopleStudioMaterials.Resource {
        try authorizePeopleMaterials(app)
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 4, let size = parts.last, ["thumb", "full"].contains(size) else {
            throw Failure("invalidRequest", "无效的素材预览地址。")
        }
        return try await peopleMaterials.sourceImage(parts.dropLast().joined(separator: "/"), full: size == "full")
    }

    private func authorizePeopleMaterials(_ app: MiniAppPackage) throws {
        try requireCurrent(app)
        guard app.builtIn && app.id == MiniAppPackage.officialID else {
            throw Failure("permissionDenied", "素材来源仅供内置人物工作室使用。")
        }
    }

    private func materialCall(_ method: String, payload: [String: Any], app: MiniAppPackage, scope: Scope) async throws -> Any {
        try authorizePeopleMaterials(app)
        switch method {
        case "materials.status":
            return try JSONSerialization.jsonObject(with: JSONEncoder().encode(await peopleMaterials.status()))
        case "materials.refresh":
            await peopleMaterials.refresh()
            return true
        case "materials.online":
            guard let enabled = payload["enabled"] as? Bool else { throw Failure("invalidRequest", "在线素材状态无效。") }
            await peopleMaterials.enableOnline(enabled)
            return enabled
        case "materials.catalog", "materials.download", "materials.cancel":
            guard let id = payload["collection"] as? String, let collection = PeopleMaterialCollection(rawValue: id) else {
                throw Failure("invalidRequest", "无效的素材分类。")
            }
            if method == "materials.catalog" {
                return try await peopleMaterials.rows(collection).map { try JSONSerialization.jsonObject(with: $0) }
            }
            if method == "materials.download" { try await peopleMaterials.startDownload(collection) }
            else { await peopleMaterials.cancelDownload(collection) }
            return true
        case "materials.claim":
            guard let id = payload["id"] as? String else { throw Failure("invalidRequest", "无效的素材标识。") }
            let image = try await peopleMaterials.sourceImage(id, full: true, forReference: true)
            let digest = SHA256.hash(data: image.data).map { String(format: "%02x", $0) }.joined()
            return try await ingest(image.data, name: image.name, bundled: "material/\(id)/\(digest)", app: app, scope: scope)
        case "materials.openSource":
            guard let id = payload["source"] as? String, let page = PeopleMaterialSources.pages[id] else {
                throw Failure("invalidRequest", "无效的素材来源入口。")
            }
            if scope.device != nil { return page.absoluteString }
            return NSWorkspace.shared.open(page)
        default:
            throw Failure("unknownMethod", "应用 SDK 没有此方法。")
        }
    }

    private func requireCurrent(_ package: MiniAppPackage) throws {
        guard self.package(package.id)?.digest == package.digest else {
            throw Failure("appChanged", "应用包已变更，请重新打开。")
        }
    }

    private func runPanel(_ panel: NSSavePanel) async throws -> Bool {
        guard !panelActive else { throw Failure("panelBusy", "请先关闭正在打开的文件选择窗口。") }
        panelActive = true
        defer { panelActive = false }
        return await withCheckedContinuation { continuation in
            if let window = NSApp.keyWindow {
                panel.beginSheetModal(for: window) { continuation.resume(returning: $0 == .OK) }
            } else {
                panel.begin { continuation.resume(returning: $0 == .OK) }
            }
        }
    }

    private func ingest(_ data: Data, name: String?, bundled: String? = nil, app: MiniAppPackage, scope: Scope) async throws -> [String: Any] {
        guard (1...Self.maxImageBytes).contains(data.count) else {
            throw Failure("mediaLimit", "单张图片上限为 16 MiB。")
        }
        var existing = try records(scope)
        if let bundled, let found = existing.first(where: { $0.value.bundledPath == bundled && $0.value.bundledDigest == app.digest }) {
            return descriptor(found.value.asset, id: found.key, scope: scope)
        }
        guard existing.count < 512, importingApps.insert(scope).inserted else {
            throw Failure("mediaLimit", "应用图片数量已达上限，或正在导入另一张图片。")
        }
        defer { importingApps.remove(scope) }
        let dir = dataDirectory(scope)
        let importer = importers[scope] ?? StudioAssetImporter(directory: dir.appendingPathComponent("media"))
        importers[scope] = importer
        let accounted = existing.values.reduce(0) { total, record in
            total + ((try? URL(fileURLWithPath: record.asset.path).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        guard data.count + accounted <= 512 * 1024 * 1024 else { throw Failure("mediaLimit", "此应用已使用 512 MiB 图片空间。") }
        var asset = try await importer.importData(data)
        var saved = false
        defer { if !saved { try? FileManager.default.removeItem(atPath: asset.path) } }
        try requireCurrent(app)
        let file = URL(fileURLWithPath: asset.path)
        let mediaType = UTType(filenameExtension: file.pathExtension)?.preferredMIMEType
        guard mediaType?.hasPrefix("image/") == true, mediaType != "image/svg+xml" else {
            throw Failure("invalidMedia", "应用只接收可解码的静态位图。")
        }
        asset.name = String(URL(fileURLWithPath: name ?? "导入图片").lastPathComponent.prefix(160))
        existing[asset.id] = MediaRecord(asset: asset, bundledPath: bundled, bundledDigest: bundled == nil ? nil : app.digest)
        try JSONEncoder().encode(existing).write(to: dir.appendingPathComponent("media.json"), options: .atomic)
        media[scope] = existing
        saved = true
        return descriptor(asset, id: asset.id, scope: scope)
    }

    func storageSnapshot(_ app: MiniAppPackage, device: UUID? = nil) -> [String: String] {
        let file = dataDirectory(Scope(app: app.id, device: device)).appendingPathComponent("storage.json")
        guard package(app.id)?.digest == app.digest, let data = try? Data(contentsOf: file), data.count <= 512 * 1024 else { return [:] }
        return (try? JSONDecoder().decode([String: String].self, from: data)) ?? [:]
    }

    private func storage(_ request: StorageRequest, scope: Scope, write: Bool) throws -> Any {
        let key = request.key
        guard (1...80).contains(key.utf8.count), MiniAppPackage.validPath(key), !key.contains("/") else {
            throw Failure("invalidRequest", "应用存储键无效。")
        }
        let dir = dataDirectory(scope)
        let file = dir.appendingPathComponent("storage.json")
        var values: [String: String] = FileManager.default.fileExists(atPath: file.path)
            ? try JSONDecoder().decode([String: String].self, from: Data(contentsOf: file)) : [:]
        if !write { return values[key].map { $0 as Any } ?? NSNull() }
        values[key] = request.value
        let data = try JSONEncoder().encode(values)
        guard values.count <= 256, data.count <= 512 * 1024 else { throw Failure("storageLimit", "应用存储空间已用完。") }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try data.write(to: file, options: .atomic)
        return true
    }

    private func authorize(_ permission: MiniAppManifest.Permission, _ app: MiniAppPackage) throws {
        guard app.manifest.permissions.contains(permission) else {
            throw Failure("permissionDenied", "未授予「\(permission.title)」权限。")
        }
    }

    private func jobInfo(_ job: NativeJob) -> [String: Any] {
        let origin = job.miniAppOrigin!
        let scope = Scope(app: origin.appID, device: origin.deviceID)
        var info: [String: Any] = ["id": job.id.uuidString, "requestID": origin.requestID.uuidString,
            "state": job.state, "phase": job.phase, "completed": job.completed, "total": job.total,
            "workflow": origin.workflow, "label": origin.label ?? "", "seed": job.request.seed,
            "model": job.request.model_variant ?? job.request.model, "createdAt": job.createdAt.timeIntervalSince1970 * 1000,
            "prompt": job.request.prompt, "referenceIDs": origin.references.map(\.uuidString),
            "dependencies": (origin.dependencies ?? []).map(\.uuidString)]
        if let context = origin.context { info["context"] = context }
        if let budget = job.request.qwen21_reference_pixel_budget {
            info["referencePixelBudget"] = budget
            if let data = job.resultJSON?.data(using: .utf8),
               let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                info["referencePixels"] = result["qwen21_reference_pixels"]
            }
        }
        if let error = job.error { info["error"] = Self.redact(error) }
        if let result = resultAsset(id: job.id, scope: scope) {
            info["output"] = descriptor(result, id: job.id, scope: scope)
        }
        return info
    }

    static func redact(_ value: String) -> String {
        String(value.prefix(2_000)).replacingOccurrences(of: #"/[^\s;,)]+"#, with: "[文件]", options: .regularExpression)
    }

    private func decode<Value: Decodable>(_ type: Value.Type, _ payload: [String: Any]) throws -> Value {
        do { return try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: payload)) }
        catch { throw Failure("invalidRequest", "应用请求参数无效。") }
    }

    func importBrowserImage(_ data: Data, name: String?, app: MiniAppPackage, device: UUID) async throws -> [String: Any] {
        try requireCurrent(app)
        try authorize(.importMedia, app)
        guard !data.isEmpty, data.count <= Self.maxImageBytes else {
            throw Failure("invalidMedia", "应用只接收不超过 16 MiB 的图片。")
        }
        let compatible = try await Task.detached { try Self.browserBitmap(data) }.value
        try Task.checkCancellation()
        try requireCurrent(app)
        return try await ingest(compatible, name: name, app: app, scope: Scope(app: app.id, device: device))
    }

    private nonisolated static func browserBitmap(_ bytes: Data) throws -> Data {
        let supported = Set(["public.png", "public.jpeg", "org.webmproject.webp", "com.compuserve.gif", "public.avif"])
        guard let source = CGImageSourceCreateWithData(bytes as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let type = CGImageSourceGetType(source) as String?,
              CGImageSourceGetCount(source) == 1,
              let info = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = info[kCGImagePropertyPixelWidth] as? Int, let height = info[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, Double(width) * Double(height) <= 80_000_000 else {
            throw Failure("invalidMedia", "请选择有效的单帧图片，限 8000 万像素。")
        }
        if supported.contains(type) { return bytes }
        // Normalize phone HEIC and other ImageIO bitmaps for Chromium/WebKit.
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1600]
        let data = NSMutableData()
        guard type != "public.svg-image", let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
              let output = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw Failure("invalidMedia", "无法将这张照片转换为网页支持的图片。")
        }
        CGImageDestinationAddImage(output, image, nil)
        guard CGImageDestinationFinalize(output) else { throw Failure("invalidMedia", "转换图片失败。") }
        return data as Data
    }

    func call(_ method: String, payload: [String: Any], app: MiniAppPackage, device: UUID? = nil,
              managesHost: Bool = false) async throws -> Any {
        try requireCurrent(app)
        let scope = Scope(app: app.id, device: device)
        if device != nil && (["models.chooseTurbo", "media.import", "media.export"].contains(method) ||
            !managesHost && ["materials.online", "materials.download", "materials.cancel"].contains(method)) {
            throw Failure("permissionDenied", "请使用此浏览器的文件操作，或在 Mac 上管理模型与素材库。")
        }
        if method.hasPrefix("materials.") { return try await materialCall(method, payload: payload, app: app, scope: scope) }
        switch method {
        case "handshake":
            return ["sdkVersion": MiniAppPackage.sdkVersion, "appID": app.id, "appVersion": app.manifest.version,
                    "permissions": app.manifest.permissions.map(\.rawValue), "maxReferences": 10, "maxBatchSize": 12,
                    "supportsDependentShots": true]
        case "storage.get", "storage.set":
            return try storage(decode(StorageRequest.self, payload), scope: scope, write: method == "storage.set")
        case "media.bundled":
            let path = try decode(PathRequest.self, payload).path
            guard let data = app.files[path], MiniAppPackage.imageExtensions.contains((path as NSString).pathExtension.lowercased()) else {
                throw Failure("mediaNotFound", "应用包中找不到这张位图。")
            }
            return try await ingest(data, name: path, bundled: path, app: app, scope: scope)
        case "media.import":
            try authorize(.importMedia, app)
            let input = try decode(DataRequest.self, payload)
            guard input.data.utf8.count <= 4 * ((Self.maxImageBytes + 2) / 3), let data = Data(base64Encoded: input.data) else {
                throw Failure("invalidMedia", "图片数据无效或超过导入上限。")
            }
            return try await ingest(data, name: input.name, app: app, scope: scope)
        case "media.get":
            let id = try decode(IDRequest.self, payload).id
            return try descriptor(asset(id: id, scope: scope), id: id, scope: scope)
        case "media.export":
            try authorize(.exportMedia, app)
            if payload["id"] != nil {
                let id = try decode(IDRequest.self, payload).id
                let (source, _) = try mediaFile(id, package: app)
                return try await save(Data(contentsOf: source), name: source.lastPathComponent)
            }
            let input = try decode(DataRequest.self, payload)
            guard input.data.utf8.count <= 4 * ((Self.maxExportBytes + 2) / 3),
                  let data = Data(base64Encoded: input.data), !data.isEmpty,
                  let name = input.name, ["png", "jpg", "jpeg", "webp", "zip", "txt", "json", "md"]
                      .contains((name as NSString).pathExtension.lowercased()) else {
                throw Failure("invalidMedia", "导出的类型无效或超过 24 MiB。")
            }
            return try await save(data, name: name)
        case "models.list":
            try authorize(.generate, app)
            return [Qwen21Installation.turboID, Qwen21Installation.modelID, "flux2-klein-4b", "z-image-turbo"].map {
                ["id": $0, "installed": (try? installation($0)) != nil] as [String: Any]
            }
        case "models.chooseTurbo":
            guard app.builtIn else { throw Failure("permissionDenied", "请在宿主模型库中管理安装。") }
            return try await chooseTurboModel()
        case "adapters.list":
            try authorize(.adapters, app)
            let selection = try installation(payload["engine"] as? String ?? Qwen21Installation.turboID)
            return try compatibleAdapters(model: selection.model, path: selection.0).map {
                ["id": $0.id, "name": $0.name, "strength": $0.defaultStrength ?? 1.0] as [String: Any]
            }
        case "workflow.plan":
            try authorize(.generate, app)
            return try await prepare(decode(PrepareRequest.self, payload), app: app, scope: scope)
        case "jobs.submit":
            try authorize(.generate, app)
            return try submit(decode(IDRequest.self, payload).id, app: app, scope: scope)
        case "jobs.get", "jobs.cancel", "jobs.remove":
            try authorize(.generate, app)
            let id = try decode(IDRequest.self, payload).id
            guard let job = store.jobs.first(where: { $0.id == id && scope.owns($0.miniAppOrigin) }) else {
                throw Failure("jobNotFound", "找不到此应用的任务。")
            }
            if method == "jobs.cancel" {
                try store.cancelGeneration(id)
                return true
            }
            if method == "jobs.remove" {
                try store.deleteJob(id)
                return true
            }
            return jobInfo(job)
        case "jobs.list":
            try authorize(.generate, app)
            return Array(store.jobs.lazy.filter { scope.owns($0.miniAppOrigin) }.prefix(200).map(jobInfo))
        default: throw Failure("unknownMethod", "应用 SDK 没有此方法。")
        }
    }

    private func compatibleAdapters(model: String, path: URL) throws -> [LibraryLoRA] {
        var target = studio.draft
        target.modelID = model
        target.modelPaths[model] = path.path
        return try (library().loras ?? []).filter {
            target.canUseLoRA($0) &&
            !StudioDraft.qwen21TurboAdapterNames.contains(URL(fileURLWithPath: $0.path).lastPathComponent)
        }
    }

    private func save(_ data: Data, name: String) async throws -> Bool {
        let panel = NSSavePanel()
        panel.title = "保存应用文件"
        let base = (name as NSString).lastPathComponent
        panel.nameFieldStringValue = base.hasPrefix(".") || base.isEmpty ? "导出文件" : String(base.prefix(160))
        guard try await runPanel(panel), let url = panel.url else { return false }
        try data.write(to: url, options: .atomic)
        return true
    }

    private func prepare(_ input: PrepareRequest, app: MiniAppPackage, scope: Scope) async throws -> [String: Any] {
        guard (1...12).contains(input.shots.count), MiniAppPackage.validPath(input.workflow),
              !input.workflow.contains("/"), input.workflow.utf8.count <= 80,
              !store.externalServiceActive, !store.requiresProcessRestart,
              planningApps.insert(scope).inserted else {
            throw Failure("notReady", "应用计划参数无效，或 GPU 正被本地 API 占用。")
        }
        defer { planningApps.remove(scope) }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let requestDigest = SHA256.hash(data: try encoder.encode(input)).map { String(format: "%02x", $0) }.joined()
        let submitted = submittedJobs(requestID: input.requestID, scope: scope)
        if !submitted.isEmpty {
            guard submitted.allSatisfy({ $0.miniAppOrigin?.requestDigest == requestDigest }) else {
                throw Failure("requestConflict", "这个提交 ID 已用于另一份请求。")
            }
            return preview(app: app, scope: scope, requestID: input.requestID, requestDigest: requestDigest,
                           jobs: submitted, estimates: submitted.map { _ in [:] })
        }
        let selection = try installation(input.engine ?? Qwen21Installation.turboID)
        let supportsWeightStreaming = selection.variant == Qwen21Installation.turboID && Self.hasConvRotSources(selection.0)
        let requestsAdapters = input.shots.contains { !($0.adapters ?? []).isEmpty }
        if requestsAdapters { try authorize(.adapters, app) }
        let registered = requestsAdapters ? try compatibleAdapters(model: selection.model, path: selection.0) : []
        var jobs: [NativeJob] = []
        var estimates: [[String: Int]] = []
        for shot in input.shots {
            guard shot.references.count <= 10 else { throw Failure("referenceLimit", "单个任务最多使用 10 张参考图。") }
            var assets: [StudioAsset] = []
            var references: [UUID] = []
            var dependencies: [UUID] = []
            for reference in shot.references {
                var bound: StudioAsset
                switch (reference.mediaID, reference.shotIndex) {
                case (let id?, nil):
                    bound = try asset(id: id, scope: scope)
                    references.append(id)
                case (nil, let index?) where jobs.indices.contains(index):
                    let parent = jobs[index]
                    let output = URL(fileURLWithPath: parent.request.output)
                    bound = StudioAsset(path: output.path, name: output.lastPathComponent, width: parent.request.width, height: parent.request.height)
                    if !dependencies.contains(parent.id) { dependencies.append(parent.id) }
                    references.append(parent.id)
                default:
                    throw Failure("invalidDependency", "参考图须引用此应用素材，或本批次中更早的任务输出。")
                }
                bound.id = UUID()
                assets.append(bound)
            }
            let adapters = shot.adapters ?? []
            guard adapters.count <= 8, Set(adapters.map(\.id)).count == adapters.count else {
                throw Failure("invalidAdapter", "同一任务最多使用 8 个不同的 LoRA。")
            }
            var adapterPaths: [NativeLoRA] = []
            for adapter in adapters {
                guard let item = registered.first(where: { $0.id == adapter.id }) else {
                    throw Failure("invalidAdapter", "适配器未登记或与当前模型不兼容。")
                }
                let strength = adapter.strength ?? item.defaultStrength ?? 1
                guard strength.isFinite, (-8...8).contains(strength) else { throw Failure("invalidAdapter", "LoRA 强度须介于 −8 和 8。") }
                if item.artifactSHA256 != nil {
                    try await Task.detached { try item.verifyArtifact() }.value
                }
                adapterPaths.append(NativeLoRA(path: item.path, strength: strength, role: item.component ?? "transformer"))
            }
            let id = UUID()
            let output = store.directory.appendingPathComponent("outputs/\(id.uuidString).png")
            let candidate = try shot.native(model: selection.model, variant: selection.variant, steps: selection.steps,
                                          output: output, assets: assets, loras: adapterPaths,
                                          supportsWeightStreaming: supportsWeightStreaming)
            let estimate = try await Self.preflight(candidate, installation: selection.0)
            estimates.append(estimate)
            jobs.append(NativeJob(id: id, createdAt: Date(), request: candidate, state: "queued", phase: "queued", completed: 0,
                                  total: candidate.steps, elapsed: 0, modelPath: selection.0.path,
                                  miniAppOrigin: MiniAppJobOrigin(appID: app.id, appVersion: app.manifest.version, appName: app.manifest.name,
                                      workflow: input.workflow, requestID: input.requestID, requestDigest: requestDigest,
                                      label: shot.label, context: shot.context, references: references, dependencies: dependencies,
                                      deviceID: scope.device),
                                  inputAssets: assets.isEmpty ? nil : assets))
        }
        try requireCurrent(app)
        return preview(app: app, scope: scope, requestID: input.requestID, requestDigest: requestDigest, jobs: jobs, estimates: estimates)
    }

    static func hasConvRotSources(_ installation: URL) -> Bool {
        let floatEncoder = installation.appendingPathComponent("text_encoders/qwen3vl_8b_bf16.safetensors")
        let int8Encoder = installation.appendingPathComponent("text_encoders/qwen3vl_8b_int8_convrot.safetensors")
        let floatAvailable = (try? floatEncoder.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        let int8Available = (try? int8Encoder.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        let int8Transformer = Qwen21Installation.checkpointName(at: installation.path) == "qwen_image_2.1_turbo_int8_convrot.safetensors"
        return floatAvailable ? int8Transformer : int8Available
    }

    private nonisolated static func preflight(_ request: NativeRequest, installation: URL) async throws -> [String: Int] {
        try await Task.detached {
            let memoryError = "qwen21_memory_budget_exceeded: "
            let data: Data
            do { data = try NativeEngine.plan(request, installation: installation) }
            catch let error as NativeFailure where error.message.hasPrefix(memoryError) {
                throw Failure("memoryBudgetExceeded", "这次生成超出本机内存预算。可以选更小画幅、减少参考图或分步编辑。\(error.message.dropFirst(memoryError.count))")
            }
            let source = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            return ["estimatedBytes": source["memory_estimate_bytes"] as? Int ?? 0,
                            "reserveBytes": source["system_reserve_bytes"] as? Int ?? 0,
                            "physicalBytes": source["physical_memory_bytes"] as? Int ?? 0]
        }.value
    }

    private func preview(app: MiniAppPackage, scope: Scope, requestID: UUID, requestDigest: String,
                         jobs: [NativeJob], estimates: [[String: Int]]) -> [String: Any] {
        let now = Date()
        plans = plans.filter { now.timeIntervalSince($0.value.created) < 600 }
        let scoped = plans.filter { $0.value.scope == scope }.sorted { $0.value.created < $1.value.created }
        for (id, _) in scoped.prefix(max(0, scoped.count - 7)) { plans.removeValue(forKey: id) }
        let id = UUID()
        plans[id] = Plan(scope: scope, appDigest: app.digest, requestID: requestID, requestDigest: requestDigest, jobs: jobs, created: now)
        return ["id": id.uuidString, "jobs": zip(jobs, estimates).enumerated().map { index, entry in
            let (job, summary) = entry
            var result: [String: Any] = ["index": index, "seed": job.request.seed,
                    "referenceCount": job.miniAppOrigin?.references.count ?? 0, "width": job.request.width,
                    "height": job.request.height, "steps": job.request.steps, "memory": summary]
            result["residency"] = job.request.residency
            if let budget = job.request.qwen21_reference_pixel_budget {
                result["referencePixelBudget"] = budget
            }
            return result
        }]
    }

    private func submittedJobs(requestID: UUID, scope: Scope) -> [NativeJob] {
        store.jobs.reversed().filter { scope.owns($0.miniAppOrigin) && $0.miniAppOrigin?.requestID == requestID }
    }

    private func submit(_ id: UUID, app: MiniAppPackage, scope: Scope) throws -> [[String: Any]] {
        guard let plan = plans[id], plan.scope == scope, plan.appDigest == app.digest,
              Date().timeIntervalSince(plan.created) < 600 else {
            throw Failure("planExpired", "生成计划已过期，请重新预检。")
        }
        let existing = submittedJobs(requestID: plan.requestID, scope: scope)
        if !existing.isEmpty {
            guard existing.allSatisfy({ $0.miniAppOrigin?.requestDigest == plan.requestDigest }) else {
                throw Failure("requestConflict", "这个提交 ID 已用于另一份请求。")
            }
            return existing.map(jobInfo)
        }
        do { try store.enqueueMiniAppJobs(plan.jobs) }
        catch { throw Failure("queueUnavailable", Self.redact(error.localizedDescription)) }
        return plan.jobs.map(jobInfo)
    }
}
