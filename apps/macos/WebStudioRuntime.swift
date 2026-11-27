import AppKit
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Browser input contains IDs and options. Only the Mac resolves them to files,
/// model installations and native requests.
@MainActor final class WebStudioRuntime {
    private struct Installation {
        let id: String
        let model: String
        let name: String
        let path: String
        let configured: Bool
    }

    private struct Request: Codable, Sendable {
        struct Adapter: Codable, Sendable {
            let id: String
            let strength: Double
        }

        let requestID: UUID
        let installation: String
        let operation: String
        let prompt: String?
        let media: [UUID]?
        let template: PlaygroundTemplate?
        let roles: [String: UUID]?
        let expansion: Double?
        let width: Int?
        let height: Int?
        let matchAspect: Bool?
        let steps: Int?
        let frames: Int?
        let fps: Int?
        let audio: Bool?
        let seed: Int?
        let strength: Double?
        let adapters: [Adapter]?
        let loraStrategy: String?
        let dynamicText: Bool?
        let residency: String?
        let streaming: StudioStreamingSelection?
        let execution: String?
        let anePartition: String?
        let compileGPU: Bool?
        let qwenReference: String?
        let qwenCache: QwenCache?
        let qwenWeightCache: String?
        let qwenWeightCacheMiB: Int?
        let qwenDiTW8A8: Bool?
        let qwenSmoothQuant: Bool?
        let promptEnhance: Bool?
        let promptEnhanceEdit: Bool?
        let tinyEncoder: Bool?
        let tinyDecoder: Bool?
        let tinyPreview: Bool?
        let tinyPreviewInterval: Int?
        let ltxBackend: String?
        let ltxVAE: String?
        let ltxSpeed: String?
        let ltxFastAV: Bool?
        let ltxBatch: Bool?
        let zImageBudget: Int?
        let seedVR2Size: String?
        let seedVR2VAE: NativeSeedVR2VAEConfig?
        let autoUpscale: String?

        enum QwenCache: String, Codable {
            case off, conservative, balanced, fast
        }
    }

    private struct IDRequest: Decodable {
        let id: UUID
    }

    private struct PreparationRequest: Decodable {
        let id: UUID
        let preset: ReferenceImagePreparation
    }

    private struct HostRequest: Decodable {
        let action: String
        let id: String?
        let model: String?
        let variant: String?
        let provider: HubProvider?
        let days: Int?
    }

    let runtime: MiniAppRuntime
    private var store: NativeJobStore { runtime.store }
    private var studio: StudioState { runtime.studio }
    private let library: ModelLibraryController
    private weak var api: LocalAPIController?
    private let monitor = ResourceMonitor()
    private var lastSample = Date.distantPast
    private let cache = TensorCacheController()
    private let thumbnails = NSCache<NSString, NSData>()
    private var hostPickerOpen = false

    init(runtime: MiniAppRuntime, library: ModelLibraryController, api: LocalAPIController?) {
        self.runtime = runtime
        self.library = library
        self.api = api
        thumbnails.totalCostLimit = 32 * 1024 * 1024
    }

    func route(_ request: LocalHTTPGateway.Request, path: String, device: LANWebServer.Device) async throws -> LocalHTTPGateway.Response {
        do {
            switch (request.method, path) {
            case ("GET", "/api/studio/catalog"):
                return try json(catalog(managing: device.canManage == true))
            case ("GET", "/api/studio/jobs"):
                let host = request.path.hasSuffix("?scope=host") && device.canManage == true
                return try json(jobs(device: device, host: host))
            case ("GET", "/api/studio/library"):
                let host = request.path.hasSuffix("?scope=host") && device.canManage == true
                let recent = visibleJobs(device: device, host: host).lazy.filter(\.hasOutput).prefix(120)
                let uploads = host ? [] : try Array(records(device.id).suffix(120).reversed()).map(media)
                return try json(["outputs": recent.map(jobInfo), "uploads": uploads])
            case ("GET", "/api/studio/device"):
                return try json(deviceStatus())
            case ("POST", "/api/studio/submit"):
                let input = try decode(Request.self, request)
                return try json(await submit(input, device: device), status: 201)
            case ("POST", "/api/studio/upload"):
                return try json(await upload(request, device: device.id), status: 201)
            case ("POST", "/api/studio/prepare"):
                return try json(await prepare(decode(PreparationRequest.self, request), device: device), status: 201)
            case ("POST", "/api/studio/cancel"), ("POST", "/api/studio/delete"),
                 ("POST", "/api/studio/trash"), ("POST", "/api/studio/reuse"):
                let input = try decode(IDRequest.self, request)
                guard let job = store.jobs.first(where: { $0.id == input.id && canRead($0, device: device) }) else {
                    throw failure(404, "找不到这项任务。")
                }
                switch path {
                case "/api/studio/cancel": try store.cancelGeneration(job.id)
                case "/api/studio/delete": try store.deleteJob(job.id)
                case "/api/studio/trash": try store.trashOutputs([job.id])
                default: return try json(reuse(job, device: device))
                }
                return try json(true)
            case ("POST", "/api/studio/host"):
                try requireManagement(device)
                return try json(await host(decode(HostRequest.self, request)))
            case ("POST", "/api/studio/host/upload"):
                try requireManagement(device)
                return try json(await uploadPackage(request), status: 201)
            default:
                let prefix = "/api/studio/media/"
                guard request.method == "GET", path.hasPrefix(prefix) else { throw failure(404, "找不到工作台接口。") }
                let parts = path.dropFirst(prefix.count).split(separator: "/", omittingEmptySubsequences: false)
                guard let first = parts.first, let id = UUID(uuidString: String(first)) else { throw failure(404, "找不到素材。") }
                let kind = parts.count == 2 ? String(parts[1]) : "file"
                guard parts.count <= 2, ["file", "preview", "source", "last"].contains(kind) else { throw failure(404, "找不到素材。") }
                return try await mediaResponse(id: id, kind: kind, device: device,
                    download: request.path == path + "?download=1", thumbnail: request.path == path + "?size=320")
            }
        } catch let error as LocalHTTPGateway.Failure {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw failure(400, MiniAppRuntime.redact(Qwen21MemoryPresentation.message(for: error)))
        }
    }

    private func failure(_ status: Int, _ message: String) -> LocalHTTPGateway.Failure {
        .init(status: status, message: message)
    }

    private func json(_ value: Any, status: Int = 200) throws -> LocalHTTPGateway.Response {
        .init(status: status, data: try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]))
    }

    private func decode<T: Decodable>(_ type: T.Type, _ request: LocalHTTPGateway.Request) throws -> T {
        let value = try LocalHTTPGateway.object(request.body)
        do { return try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: value)) }
        catch { throw failure(400, "请求的参数或素材格式无效。") }
    }

    private func installIndex() throws -> (LibraryIndex, [Installation]) {
        let index = try LibraryStore(createIfMissing: false).read()
        var candidates = index.installations.map { item in
            Installation(id: item.id, model: item.modelID, name: item.name, path: item.path, configured: false)
        }
        var configured = studio.draft.modelPaths
        for variant in UpscaleVariant.allCases {
            configured[variant.modelID] = studio.draft.upscaleModelPaths[variant.rawValue]
                ?? (studio.draft.upscaleVariant == variant ? studio.draft.upscaleModelPath : "")
        }
        for (model, path) in configured.sorted(by: { $0.key < $1.key }) where !path.isEmpty {
            if let position = candidates.firstIndex(where: { $0.model == model && $0.path == path }) {
                let item = candidates.remove(at: position)
                candidates.insert(.init(id: item.id, model: model, name: item.name, path: path, configured: true), at: 0)
            } else {
                let id = "configured-" + SHA256.hash(data: Data((model + "\n" + path).utf8)).description
                let name = URL(fileURLWithPath: path).lastPathComponent
                candidates.insert(.init(id: id, model: model, name: name, path: path, configured: true), at: 0)
            }
        }
        return (index, candidates.filter { FileManager.default.isReadableFile(atPath: $0.path) })
    }

    private func modelDraft(_ model: StudioModel, installation: Installation) -> StudioDraft {
        var draft = StudioDraft()
        draft.applyModelDefaults(model)
        draft.modelPaths[model.id] = installation.path
        draft.ltxPrunaVAEPath = library.prunaVAEPath ?? studio.draft.ltxPrunaVAEPath
        draft.tinyVAEPaths = studio.draft.tinyVAEPaths
        for asset in TinyVAEAsset.all {
            if draft.tinyVAEPaths[asset.id] == nil { draft.tinyVAEPaths[asset.id] = library.tinyVAEPath(asset) }
        }
        draft.promptEnhancerPath = studio.draft.promptEnhancerPath
        draft.normalizeZImageResidency()
        draft.normalizeQwen21CheckpointRecipe()
        return draft
    }

    private func catalog(managing: Bool) throws -> [String: Any] {
        let (index, installations) = try installIndex()
        var models: [[String: Any]] = studio.models.filter(\.executor).map { model in
            let installRows: [[String: Any]] = installations.filter { $0.model == model.id }.map { item in
                let draft = modelDraft(model, installation: item)
                let loras = (index.loras ?? []).filter { draft.canUseLoRA($0) }
                return ["id": item.id, "name": item.name, "configured": item.configured, "turbo": draft.hasQwen21TurboCheckpoint,
                        "convrot": draft.hasQwen21ConvrotCheckpoint, "residency": draft.residency,
                        "pruna": draft.resolvedPrunaVAEPath != nil, "steps": draft.steps,
                        "loras": loras.map(loraInfo)]
            }
            let variants = LibraryRecipe.all.first { $0.modelID == model.id }?.downloadVariants ?? []
            let square = !model.isVideo || model.id == "ltx-2.5-distilled"
            return ["id": model.id, "name": model.displayName, "kind": model.output,
                    "operations": model.availableOperations, "steps": model.default_steps,
                    "width": square ? 512 : model.default_width, "height": square ? 512 : model.default_height,
                    "frames": model.default_frames, "fps": model.default_fps ?? (model.isVideo ? 24 : 1),
                    "audio": model.canGenerateAudio, "defaultAudio": model.default_audio ?? false,
                    "maxImages": model.max_images ?? 8, "supportsANE": model.supports_gpu_ane ?? false,
                    "supportsLoRA": model.supports_lora ?? false, "strategies": model.lora_strategies ?? [],
                    "installations": installRows, "variants": variants.map(variantInfo)]
        }
        for variant in UpscaleVariant.allCases where !models.contains(where: { $0["id"] as? String == variant.modelID }) {
            models.append(["id": variant.modelID, "name": variant.title, "kind": "image",
                           "operations": ["image.upscale"], "coreML": true, "download": variant.downloadURL.absoluteString,
                           "installations": installations.filter { $0.model == variant.modelID }.map {
                ["id": $0.id, "name": $0.name, "configured": $0.configured, "loras": []] as [String: Any]
            }, "variants": []])
        }
        let templates = PlaygroundTemplate.allCases.map { template in
            ["id": template.rawValue, "name": template.title, "detail": template.detail, "instruction": template.defaultInstruction,
             "roles": template.roles.map { role in
                ["id": role.rawValue, "title": template.title(for: role),
                 "required": template.requiredRoles.contains(role)] as [String: Any]
             }] as [String: Any]
        }
        let partitions = (index.anePartitions ?? []).map { partition in
            ["id": partition.id, "model": partition.modelID, "name": partition.capacity + " · " + partition.kind, "kind": partition.kind]
        }
        let vaes = TinyVAEAsset.all.map { asset in
            ["id": asset.id, "name": asset.title, "installed": library.tinyVAEPath(asset) != nil || asset.bundledPath() != nil] as [String: Any]
        }
        var response: [String: Any] = ["models": models, "templates": templates, "ane": partitions,
            "tinyVAE": vaes, "pruna": library.prunaVAEPath != nil, "preferred": studio.draft.modelID, "canManage": managing]
        if managing {
            response["registeredLoRAs"] = (index.loras ?? []).map(loraInfo)
        }
        return response
    }

    private func loraInfo(_ lora: LibraryLoRA) -> [String: Any] {
        ["id": lora.id, "name": lora.name, "strength": lora.defaultStrength ?? 1.0, "models": lora.modelIDs,
         "viggle": StudioDraft.qwen21TurboAdapterNames.contains(URL(fileURLWithPath: lora.path).lastPathComponent)]
    }

    private func variantInfo(_ variant: LibraryVariant) -> [String: Any] {
        ["id": variant.id, "name": variant.title, "provider": variant.pinnedProvider?.rawValue ?? "",
         "runnable": variant.runnable, "preparation": variant.preparation]
    }

    private func deviceStatus() -> [String: Any] {
        if Date().timeIntervalSince(lastSample) >= 2 { monitor.sample(); lastSample = Date() }
        var result: [String: Any] = [
            "queue": store.jobs.lazy.filter { $0.state == "queued" }.count,
            "busy": store.busy, "canUnload": store.canUnload, "requiresRestart": store.requiresProcessRestart,
            "memory": ProcessInfo.processInfo.physicalMemory, "apiRunning": api?.running == true,
            "apiPort": api?.httpPort ?? 0, "apiAvailable": api != nil,
            "download": library.busy, "downloadMessage": MiniAppRuntime.redact(library.message ?? ""),
            "cacheBusy": cache.busy, "cacheDays": cache.retentionDays, "cacheMessage": MiniAppRuntime.redact(cache.message ?? "")
        ]
        result["loadedModel"] = store.loadedModelID
        result["cpu"] = monitor.cpuPercent
        result["gpu"] = monitor.gpuPercent
        result["resident"] = monitor.residentBytes
        if let report = cache.report { result["cacheBytes"] = report.bytes; result["cacheEntries"] = report.entries.count }
        if let event = library.event, library.busy {
            result["transfer"] = ["phase": event.phase, "completed": event.completedBytes, "total": event.totalBytes]
        }
        return result
    }

    private func workspace(_ device: UUID) -> URL {
        store.directory.appendingPathComponent("lan-web/workspaces/\(device.uuidString)", isDirectory: true)
    }

    private func records(_ device: UUID) throws -> [StudioAsset] {
        let file = workspace(device).appendingPathComponent("assets.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return [] }
        return try JSONDecoder().decode([StudioAsset].self, from: Data(contentsOf: file))
    }

    private func save(_ asset: StudioAsset, device: UUID) throws {
        let folder = workspace(device)
        var updated = try records(device)
        guard updated.count < 2_000 else { throw failure(429, "素材库已达上限。") }
        updated.append(asset)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        try JSONEncoder().encode(updated).write(to: folder.appendingPathComponent("assets.json"), options: .atomic)
    }

    private func importer(_ id: UUID) -> StudioAssetImporter {
        StudioAssetImporter(directory: workspace(id).appendingPathComponent("assets", isDirectory: true))
    }

    private func upload(_ request: LocalHTTPGateway.Request, device: UUID) async throws -> [String: Any] {
        guard let file = request.bodyFile else { throw failure(400, "没有收到素材文件。") }
        let name = uploadName(request)
        let type = request.headers["x-turbocider-kind"]
        let receiver = importer(device)
        var asset: StudioAsset
        if type == "video" {
            let ext = (name as NSString).pathExtension.lowercased()
            guard ["mp4", "mov", "m4v"].contains(ext) else { throw failure(400, "请选择 MP4、MOV 或 M4V 视频。") }
            let typed = file.appendingPathExtension(ext)
            try FileManager.default.linkItem(at: file, to: typed)
            defer { try? FileManager.default.removeItem(at: typed) }
            asset = try await receiver.importVideoFile(typed, preserveTiming: true, selfContained: true)
        } else {
            guard type == "image", (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= 32 * 1024 * 1024 else {
                throw failure(413, "请上传 32 MiB 以内的单张图片。")
            }
            asset = try await receiver.importFile(file)
        }
        asset.name = name
        asset.original?.name = name
        do { try Task.checkCancellation(); try save(asset, device: device) }
        catch { await receiver.discard([asset]); throw error }
        return media(asset)
    }

    private func uploadName(_ request: LocalHTTPGateway.Request) -> String {
        let name = request.headers["x-turbocider-filename"]?.removingPercentEncoding ?? "素材"
        return String(URL(fileURLWithPath: name).lastPathComponent.filter { !$0.isNewline && $0 != "\0" }.prefix(160))
    }

    private func prepare(_ request: PreparationRequest, device: LANWebServer.Device) async throws -> [String: Any] {
        var original = try asset(request.id, device: device)
        guard original.mediaKind == "image" else { throw failure(400, "只有图片可以预处理。") }
        original.id = UUID()
        let reader = importer(device.id)
        let prepared = try await reader.prepare(asset: original, preset: request.preset)
        do { try Task.checkCancellation(); try save(prepared.asset, device: device.id) }
        catch { await reader.discardPreparations([prepared]); throw error }
        return media(prepared.asset)
    }

    private func canRead(_ job: NativeJob, device: LANWebServer.Device) -> Bool {
        owns(job, device: device.id) || (device.canManage == true && isHost(job))
    }

    private func isHost(_ job: NativeJob) -> Bool {
        job.webOrigin == nil && job.miniAppOrigin?.deviceID == nil
    }

    private func owns(_ job: NativeJob, device: UUID) -> Bool {
        job.webOrigin?.deviceID == device || job.miniAppOrigin?.deviceID == device
    }

    private func visibleJobs(device: LANWebServer.Device, host: Bool) -> [NativeJob] {
        store.jobs.filter { host ? isHost($0) : owns($0, device: device.id) }
    }

    private func jobs(device: LANWebServer.Device, host: Bool) -> [String: Any] {
        let visible = visibleJobs(device: device, host: host)
        let active = visible.filter { !$0.isTerminal }
        let recent = visible.lazy.filter(\.isTerminal).prefix(120)
        return ["items": (active + Array(recent)).map(jobInfo), "queued": store.jobs.lazy.filter { $0.state == "queued" }.count,
                "busy": store.busy]
    }

    private func media(_ asset: StudioAsset) -> [String: Any] {
        let url = "/api/studio/media/\(asset.id.uuidString)"
        var result: [String: Any] = ["id": asset.id.uuidString, "name": asset.name,
            "kind": asset.mediaKind, "width": asset.width, "height": asset.height,
            "url": url, "preview": asset.mediaKind == "video" ? url : url + "/preview"]
        result["frames"] = asset.frames
        result["fps"] = asset.fps
        return result
    }

    private func jobInfo(_ job: NativeJob) -> [String: Any] {
        let request = job.request
        var result: [String: Any] = [
            "id": job.id.uuidString, "model": request.model_variant ?? request.model, "operation": request.operation ?? "",
            "label": job.miniAppOrigin?.label ?? job.workflowName ?? "",
            "prompt": String(request.prompt.prefix(12_000)), "seed": request.seed, "created": job.createdAt.timeIntervalSince1970 * 1_000,
            "state": job.state, "phase": job.phase, "completed": job.completed, "total": job.total, "elapsed": job.elapsed,
            "width": request.width, "height": request.height, "frames": request.frames,
            "steps": request.steps, "source": job.automationRequestJSON != nil ? "API" : job.miniAppOrigin != nil ? "应用" : job.webOrigin != nil ? "网页" : "Mac"
        ]
        result["fps"] = request.fps
        result["route"] = job.routeSummary
        result["error"] = job.error.map(MiniAppRuntime.redact)
        if job.state == "queued" {
            let position = store.jobs.reversed().lazy.filter { $0.state == "queued" }.prefix { $0.id != job.id }.count
            result["position"] = position + 1
        }
        if let output = outputAsset(job) {
            result["output"] = media(output)
            if let source = inputFile(job, last: false) {
                result["comparison"] = ["url": "/api/studio/media/\(job.id.uuidString)/source", "kind": StudioMediaKind(path: source.path).rawValue]
            }
            if inputFile(job, last: true) != nil { result["lastFrame"] = "/api/studio/media/\(job.id.uuidString)/last" }
        }
        if store.activeJob?.id == job.id, store.livePreview != nil {
            result["livePreview"] = "/api/studio/media/\(job.id.uuidString)/preview?step=\(store.livePreviewStep)"
        }
        return result
    }

    private func outputAsset(_ job: NativeJob) -> StudioAsset? {
        guard job.hasOutput, let file = ownedFile(job.request.output, in: "outputs") else { return nil }
        return StudioAsset(id: job.id, path: file.path, name: "TurboCider-\(job.id.uuidString).\(file.pathExtension)",
                           width: job.request.width, height: job.request.height,
                           kind: job.outputMediaKind.rawValue, frames: job.request.frames, fps: job.request.fps)
    }

    private func ownedFile(_ path: String, in folder: String? = nil) -> URL? {
        let root = folder.map { store.directory.appendingPathComponent($0) } ?? store.directory
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
        guard url.path.hasPrefix(root.resolvingSymlinksInPath().standardizedFileURL.path + "/"),
              (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
              FileManager.default.isReadableFile(atPath: url.path) else { return nil }
        return url
    }

    private func inputFile(_ job: NativeJob, last: Bool) -> URL? {
        let inputs = job.request.inputs ?? []
        let input = last ? inputs.first { $0.role == "last_frame" } : inputs.first
        guard let input else { return nil }
        return ownedFile(input.path)
    }

    private func asset(_ id: UUID, device: LANWebServer.Device) throws -> StudioAsset {
        if let asset = try records(device.id).first(where: { $0.id == id }), ownedFile(asset.path) != nil { return asset }
        if let job = store.jobs.first(where: { $0.id == id && canRead($0, device: device) }), let asset = outputAsset(job) { return asset }
        throw failure(404, "找不到这个工作区的素材。")
    }

    private func mediaResponse(id: UUID, kind: String, device: LANWebServer.Device, download: Bool,
                               thumbnail: Bool) async throws -> LocalHTTPGateway.Response {
        let job = store.jobs.first { $0.id == id && canRead($0, device: device) }
        if kind == "preview", let job, store.activeJob?.id == job.id, let current = store.livePreview {
            return .init(status: 200, data: current, mime: "image/png")
        }
        if let job, job.hasOutput, kind == "source" || kind == "last" {
            guard let source = inputFile(job, last: kind == "last") else { throw failure(404, "找不到输入素材。") }
            return try await preview(source)
        }
        guard kind == "file" || kind == "preview" else { throw failure(404, "找不到素材。") }
        let asset = try asset(id, device: device)
        let file = URL(fileURLWithPath: asset.path)
        if kind == "preview" && asset.mediaKind == "image" { return try await preview(file, maxPixel: thumbnail ? 320 : 1440) }
        let mime = UTType(filenameExtension: file.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
        let allowed = (mime.hasPrefix("image/") && mime != "image/svg+xml") || mime.hasPrefix("video/")
        guard allowed else { throw failure(415, "浏览器不支持此媒体类型。") }
        let ext = file.pathExtension.lowercased().filter { $0.isASCII && $0.isLetter || $0.isASCII && $0.isNumber }
        let headers = download ? ["Content-Disposition": "attachment; filename=\"TurboCider-\(id.uuidString).\(ext)\""] : [:]
        return .init(status: 200, data: Data(), mime: mime, file: file, headers: headers, ranges: asset.mediaKind == "video")
    }

    private func preview(_ file: URL, maxPixel: Int = 1440) async throws -> LocalHTTPGateway.Response {
        if StudioMediaKind(path: file.path) == .video {
            let mime = UTType(filenameExtension: file.pathExtension)?.preferredMIMEType ?? "video/mp4"
            return .init(status: 200, data: Data(), mime: mime, file: file, ranges: true)
        }
        let key = "\(maxPixel)\n\(file.path)" as NSString
        if let cached = thumbnails.object(forKey: key) { return .init(status: 200, data: cached as Data, mime: "image/png") }
        let data = try await Task.detached(priority: .utility) { try Self.thumbnail(file, maxPixel: maxPixel) }.value
        thumbnails.setObject(data as NSData, forKey: key, cost: data.count)
        return .init(status: 200, data: data, mime: "image/png")
    }

    private nonisolated static func thumbnail(_ file: URL, maxPixel: Int) throws -> Data {
        let output = NSMutableData()
        guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixel
              ] as CFDictionary),
              let writer = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else {
            throw NativeFailure(message: "无法生成素材预览。")
        }
        CGImageDestinationAddImage(writer, image, nil)
        guard CGImageDestinationFinalize(writer) else { throw NativeFailure(message: "无法生成素材预览。") }
        return output as Data
    }

    private func reuse(_ job: NativeJob, device: LANWebServer.Device) throws -> [String: Any] {
        if let raw = job.webOrigin?.recipe, let value = try JSONSerialization.jsonObject(with: raw) as? [String: Any] {
            var replay = value
            replay.removeValue(forKey: "requestID")
            replay["seed"] = job.request.seed
            return replay
        }
        let request = job.request
        var result: [String: Any] = ["prompt": request.prompt, "operation": request.operation ?? "image.generate", "width": request.width,
            "height": request.height, "steps": request.steps, "seed": request.seed, "frames": request.frames]
        result["fps"] = request.fps
        result["audio"] = request.audio
        result["installation"] = try installIndex().1.first { $0.path == job.modelPath && $0.model == request.model }?.id
        // Import trusted App input copies into this workspace before recreating bindings.
        var ids: [String] = []
        for input in request.inputs ?? [] {
            guard input.kind == "image" || input.kind == "video", let file = ownedFile(input.path) else { continue }
            let existing = try records(device.id).first { $0.path == file.path }
            var source = job.inputAssets?.first { $0.path == input.path }
                ?? StudioAsset(path: file.path, name: file.lastPathComponent, width: 0, height: 0, kind: input.kind)
            if source.width <= 0 && input.kind == "image",
               let reader = CGImageSourceCreateWithURL(file as CFURL, nil),
               let props = CGImageSourceCopyPropertiesAtIndex(reader, 0, nil) as? [CFString: Any] {
                source.width = props[kCGImagePropertyPixelWidth] as? Int ?? 0
                source.height = props[kCGImagePropertyPixelHeight] as? Int ?? 0
            }
            if let existing { ids.append(existing.id.uuidString) }
            else { source.id = UUID(); try save(source, device: device.id); ids.append(source.id.uuidString) }
        }
        result["media"] = ids
        return result
    }

    private func submit(_ input: Request, device: LANWebServer.Device) async throws -> [[String: Any]] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let recipe = try encoder.encode(input)
        let digest = SHA256.hash(data: recipe).description
        if let existing = try priorSubmission(input.requestID, device: device.id, digest: digest) { return existing }
        let (index, installations) = try installIndex()
        guard let installation = installations.first(where: { $0.id == input.installation }) else {
            throw failure(404, "所选模型安装不可用，请在模型中心选择或下载模型。")
        }
        let mediaIDs = input.media ?? []
        guard mediaIDs.count <= 10, Set(mediaIDs).count == mediaIDs.count,
              (input.prompt?.utf8.count ?? 0) <= 12_000, input.prompt?.contains("\0") != true else {
            throw failure(400, "请检查提示词和参考图数量。")
        }
        let origin = WebJobOrigin(deviceID: device.id, requestID: input.requestID, digest: digest, mediaIDs: mediaIDs,
                                  template: input.template?.rawValue, afterJob: nil, recipe: recipe)
        let assets = try mediaIDs.map { try asset($0, device: device) }
        let native: NativeJob
        if UpscaleVariant.from(modelID: installation.model) != nil {
            guard input.operation == "image.upscale", assets.count == 1, let source = assets.first, source.mediaKind == "image",
                  let compute = UpscaleCompute(rawValue: input.execution ?? "gpu") else {
                throw failure(400, "Real-ESRGAN 需要一张源图片和有效的计算设置。")
            }
            native = try coreMLJob(installation, source: source, compute: compute, origin: origin)
        } else {
            guard let model = studio.models.first(where: { $0.id == installation.model && $0.executor }),
                  model.supports(input.operation) || input.template != nil else {
                throw failure(400, "当前模型不支持此创作方式。")
            }
            var draft = try configure(input, model: model, installation: installation, index: index, assets: assets, device: device)
            let id = UUID()
            let output = outputURL(id, kind: draft.operation.hasPrefix("video.") ? "mp4" : "png")
            let referenceMode = input.qwenReference ?? (draft.hasQwen21TurboCheckpoint ? "budget" : "standard")
            let usesBudget = draft.hasQwen21TurboCheckpoint && draft.operation == "image.edit" && referenceMode == "budget"
            if usesBudget && Double(draft.width) * Double(draft.height) > Double(768 * 512) {
                throw failure(400, "Turbo 的共享参考图预算适用于总像素不超过 768×512 的画布。")
            }
            // Acceleration discovery is shared with Studio. It reads only the
            // model/partition paths resolved by the host.
            draft = try await store.resolveAcceleration(draft)
            let resolved = draft
            var pair = try await Task.detached { try resolved.publicStreamingRequest(output: output) }.value
            if usesBudget {
                pair.legacy.qwen21_reference_size = nil
                pair.legacy.qwen21_reference_pixel_budget = 512 * 512
                if input.residency != "resident", resolved.activeLoRAs.isEmpty,
                   MiniAppRuntime.hasConvRotSources(URL(fileURLWithPath: installation.path)) {
                    pair.legacy.residency = "streamed"
                }
            }
            let runtimeOptions = try resolved.runtimeOptions(store: store.directory)
            let modelURL = URL(fileURLWithPath: installation.path)
            if pair.v2 == nil && runtimeOptions == nil {
                let request = pair.legacy
                _ = try await Task.detached { try NativeEngine.plan(request, installation: modelURL) }.value
            }
            let intent = try pair.v2.map { String(decoding: try encoder.encode($0), as: UTF8.self) }
            native = NativeJob(id: id, createdAt: Date(), request: pair.legacy, state: "queued", phase: "queued", completed: 0,
                total: pair.legacy.steps, elapsed: 0, modelPath: installation.path,
                publicStreamingTargetBytes: pair.v2?.execution.streaming?.target_request_memory_bytes,
                publicStreamingIntentJSON: intent, runtimeOptions: runtimeOptions,
                workflowID: input.template?.workflowID, webOrigin: origin,
                inputAssets: NativeJob.matchingInputAssets(resolved.activeAssets, for: pair.legacy))
        }

        var batch = [native]
        if let upscaler = input.autoUpscale {
            guard native.request.operation?.hasPrefix("image.") == true, native.request.operation != "image.upscale",
                  let selected = installations.first(where: { $0.id == upscaler }),
                  UpscaleVariant.from(modelID: selected.model) != nil else {
                throw failure(400, "生成后超分需要已安装的 Real-ESRGAN 图像模型。")
            }
            let source = StudioAsset(id: native.id, path: native.request.output, name: "创作结果",
                                     width: native.request.width, height: native.request.height)
            let child = WebJobOrigin(deviceID: device.id, requestID: input.requestID, digest: digest,
                                     mediaIDs: [native.id], template: nil, afterJob: native.id)
            batch.append(try coreMLJob(selected, source: source, compute: .gpu, origin: child))
        }

        try Task.checkCancellation()
        if let existing = try priorSubmission(input.requestID, device: device.id, digest: digest) { return existing }
        try store.enqueueGenerationJobs(batch)
        return batch.map(jobInfo)
    }

    private func priorSubmission(_ id: UUID, device: UUID, digest: String) throws -> [[String: Any]]? {
        let matches = store.jobs.filter { $0.webOrigin?.deviceID == device && $0.webOrigin?.requestID == id }
        guard !matches.isEmpty else { return nil }
        guard matches.allSatisfy({ $0.webOrigin?.digest == digest }) else { throw failure(409, "请求编号已经用于其他创作。") }
        return matches.reversed().map(jobInfo)
    }

    private func outputURL(_ id: UUID, kind: String) -> URL {
        store.directory.appendingPathComponent("outputs/\(id.uuidString).\(kind)")
    }

    private func coreMLJob(_ installation: Installation, source: StudioAsset, compute: UpscaleCompute,
                           origin: WebJobOrigin) throws -> NativeJob {
        guard let variant = UpscaleVariant.from(modelID: installation.model) else { throw failure(400, "无效的 Core ML 模型。") }
        try ImageUpscaler.validateModelURL(URL(fileURLWithPath: installation.path))
        if origin.afterJob == nil { _ = try ImageUpscaler.dimensions(URL(fileURLWithPath: source.path), scale: variant.scale) }
        let id = UUID()
        var request = NativeRequest(prompt: "图像超分 ×\(variant.scale)", output: outputURL(id, kind: "png").path)
        request.model = installation.model; request.operation = "image.upscale"
        request.width = source.width * variant.scale; request.height = source.height * variant.scale
        request.steps = 1; request.seed = 0; request.execution = compute.rawValue
        request.inputs = [NativeInput(kind: "image", role: "source", path: source.path)]
        return NativeJob(id: id, createdAt: Date(), request: request, state: "queued", phase: "queued",
                         completed: 0, total: 1, elapsed: 0, modelPath: installation.path, webOrigin: origin, inputAssets: [source])
    }

    private func configure(_ input: Request, model: StudioModel, installation: Installation, index: LibraryIndex,
                           assets: [StudioAsset], device: LANWebServer.Device) throws -> StudioDraft {
        var draft = modelDraft(model, installation: installation)
        draft.operation = input.operation
        draft.prompt = input.prompt ?? ""
        draft.assets = assets
        if let template = input.template {
            guard model.id == Qwen21Installation.modelID, input.autoUpscale == nil,
                  let roles = input.roles, Set(roles.keys).isSubset(of: Set(template.roles.map(\.rawValue))) else {
                throw failure(400, "请选择 Qwen Image 2.1 和此模板接受的参考图。")
            }
            for role in template.requiredRoles where roles[role.rawValue] == nil {
                throw failure(400, "请添加\(template.title(for: role))参考图。")
            }
            let bound = try template.roles.compactMap { role -> (String, StudioAsset)? in
                guard let id = roles[role.rawValue] else { return nil }
                let image = try asset(id, device: device)
                guard image.mediaKind == "image" else { throw failure(400, "模板参考素材需为图片。") }
                return (role.rawValue, image)
            }
            let compiled = try template.compile(paths: Dictionary(uniqueKeysWithValues: bound.map { ($0.0, $0.1.path) }),
                                                instruction: draft.prompt, modelID: model.id, expansion: input.expansion ?? 1.5)
            draft.assets = bound.map(\.1)
            draft.operation = compiled.operation; draft.prompt = compiled.prompt
        }
        let expectedKind = draft.operation == "video.upscale" ? "video" : "image"
        guard draft.assets.allSatisfy({ $0.mediaKind == expectedKind }) else { throw failure(400, "素材类型与工作流不匹配。") }
        if ["image.generate", "video.generate"].contains(draft.operation), !draft.assets.isEmpty {
            throw failure(400, "纯文本生成无需参考素材，请切换到参考编辑或图生视频。")
        }
        if ["image.upscale", "video.upscale", "image.transform", "video.image"].contains(draft.operation), draft.assets.count != 1 {
            throw failure(400, "此工作流需要一个源素材。")
        }
        if draft.operation == "video.keyframes", draft.assets.count != 2 {
            throw failure(400, "请依次添加首帧和尾帧。")
        }
        draft.initImageID = draft.assets.first?.id
        draft.lastImageID = draft.assets.dropFirst().first?.id
        draft.width = input.width ?? draft.width; draft.height = input.height ?? draft.height
        if input.matchAspect == true, let image = draft.assets.first, image.width > 0, image.height > 0, !draft.isNativeUpscale {
            let area = Double(image.width) * Double(image.height)
            let scale = min(2048 / Double(max(image.width, image.height)), sqrt(Double(512 * 512) / area))
            let multiple = model.id == "ltx-2.5-distilled" ? 64 : model.id == Qwen21Installation.modelID ? 32 : 16
            let dimensions = EditingCanvasSizing.alignedDimensions(.init(width: image.width, height: image.height),
                                                                   scale: scale, multiple: multiple, shrinkOnly: false)
            draft.width = max(64, dimensions.width); draft.height = max(64, dimensions.height)
        }
        draft.steps = input.steps ?? draft.steps; draft.frames = input.frames ?? draft.frames
        draft.fps = input.fps ?? draft.fps; draft.audio = input.audio ?? draft.audio
        draft.randomSeed = input.seed == nil
        if let seed = input.seed { draft.seedText = String(seed) }
        draft.strength = input.strength ?? (draft.operation == "video.keyframes" ? 1 : draft.strength)
        draft.dynamicText = input.dynamicText ?? draft.dynamicText

        let adapters = input.adapters ?? []
        guard adapters.count <= 8, Set(adapters.map(\.id)).count == adapters.count, model.supports_lora == true || adapters.isEmpty else {
            throw failure(400, "所选模型不支持这组 LoRA。")
        }
        for adapter in adapters {
            guard let lora = (index.loras ?? []).first(where: { $0.id == adapter.id && draft.canUseLoRA($0) }),
                  adapter.strength.isFinite, (-8...8).contains(adapter.strength) else {
                throw failure(400, "LoRA 与安装不兼容，或强度超出 −8 到 8。")
            }
            draft.loras.append(StudioLoRA(path: lora.path, strength: adapter.strength, role: lora.component ?? "transformer", registration: lora))
        }
        if let strategy = input.loraStrategy {
            guard strategy == "auto" || model.lora_strategies?.contains(strategy) == true else { throw failure(400, "LoRA 策略不可用。") }
            draft.loraStrategy = strategy
        }
        draft.normalizeQwen21CheckpointRecipe(preferStreaming: input.residency != "resident")
        if let residency = input.residency {
            guard ["resident", "component_staged", "streamed"].contains(residency) else { throw failure(400, "加载方式无效。") }
            draft.residency = residency
        }
        if let selection = input.streaming {
            draft.streaming.selection = selection
            draft.streaming.userSelected = true
        }
        if let execution = input.execution {
            guard ["gpu", "auto", "gpu_ane"].contains(execution), execution != "gpu_ane" || model.supports_gpu_ane == true else {
                throw failure(400, "计算设备与模型不兼容。")
            }
            draft.acceleration?.policy = execution
        }
        if let partition = input.anePartition {
            guard let item = (index.anePartitions ?? []).first(where: { $0.id == partition && $0.modelID == model.id }) else {
                throw failure(400, "所选 ANE 分区不可用。")
            }
            if item.kind == "compiled" { draft.acceleration?.manifest = item.path }
            else { draft.acceleration?.sourceManifest = item.path }
        }
        draft.acceleration?.compileGPU = input.compileGPU
        draft.promptEnhance = input.promptEnhance ?? false
        draft.promptEnhanceEditExperimental = input.promptEnhanceEdit ?? false
        draft.tinyVAEEncoder = input.tinyEncoder == true ? "tiny" : "original"
        draft.tinyVAEDecoder = input.tinyDecoder == true ? "tiny" : "original"
        draft.tinyPreview = input.tinyPreview ?? false
        draft.tinyPreviewInterval = input.tinyPreviewInterval ?? draft.tinyPreviewInterval

        if model.id == Qwen21Installation.modelID {
            if let reference = input.qwenReference {
                guard ["budget", "standard", "fast"].contains(reference), reference != "budget" || draft.hasQwen21TurboCheckpoint else {
                    throw failure(400, "仅八步 Turbo 支持参考图共享像素预算。")
                }
                draft.qwen21ReferenceSize = reference == "fast" ? 512 : 1024
            }
            draft.qwen21DiTCache = input.qwenCache?.rawValue ?? draft.qwen21DiTCache
            draft.qwen21WeightCacheBudget = input.qwenWeightCache ?? draft.qwen21WeightCacheBudget
            draft.qwen21WeightCacheCustomMiB = input.qwenWeightCacheMiB ?? draft.qwen21WeightCacheCustomMiB
            draft.qwen21DiTW8A8 = input.qwenDiTW8A8 ?? false
            draft.qwen21DiTSmoothQuantConvRot = input.qwenSmoothQuant ?? false
        }
        draft.zImageStreamingBudgetGiB = input.zImageBudget ?? draft.zImageStreamingBudgetGiB
        if model.id == "ltx-2.5-distilled" {
            draft.ltxBackend = input.ltxBackend ?? (draft.operation == "video.keyframes" ? "c_metal" : draft.ltxBackend)
            draft.ltxVideoVAE = input.ltxVAE ?? draft.ltxVideoVAE
            draft.ltxAccelerationMode = input.ltxSpeed ?? draft.ltxAccelerationMode
            draft.ltxFastAV = input.ltxFastAV ?? draft.ltxFastAV
            draft.ltxVideoAttentionBatch = input.ltxBatch ?? draft.ltxVideoAttentionBatch
        }
        if model.id == "seedvr2-3b", let source = draft.activeAssets.first {
            draft.seedvr2VAE = input.seedVR2VAE
            if input.width == nil || input.height == nil || input.seedVR2Size != nil {
                let suggested = (try? SeedVR2OutputPreset.scale2.dimensions(source: source))
                    ?? (try? SeedVR2OutputPreset.long1024.dimensions(source: source))
                let dimensions = try input.seedVR2Size.flatMap(SeedVR2OutputPreset.init(rawValue:))?.dimensions(source: source) ?? suggested
                if let dimensions { draft.width = dimensions.width; draft.height = dimensions.height }
            }
            if draft.operation == "video.upscale" {
                guard let count = source.frames, let rate = source.fps else { throw failure(400, "读取不到源视频帧数或帧率。") }
                draft.frames = count; draft.fps = rate
            } else { draft.frames = 1 }
        }
        try draft.validate(model: model)
        return draft
    }

    private func requireManagement(_ device: LANWebServer.Device) throws {
        guard device.canManage == true else { throw failure(403, "请在 Mac 上给已配对设备授权主机管理。") }
    }

    private func uploadPackage(_ request: LocalHTTPGateway.Request) async throws -> [String: Any] {
        guard let source = request.bodyFile, !library.busy else { throw failure(409, "请等待模型库操作结束。") }
        let name = uploadName(request)
        let ext = (name as NSString).pathExtension.lowercased()
        switch request.headers["x-turbocider-kind"] {
        case "app" where ext == "tcapp":
            let typed = source.appendingPathExtension("tcapp")
            try FileManager.default.linkItem(at: source, to: typed)
            defer { try? FileManager.default.removeItem(at: typed) }
            let package = try await Task.detached { try MiniAppPackage(source: typed) }.value
            try Task.checkCancellation()
            try runtime.install(package)
            return ["name": package.manifest.name]
        case "lora" where ext == "safetensors":
            guard let model = request.headers["x-turbocider-model"],
                  studio.models.contains(where: { $0.id == model && $0.supports_lora == true }) else {
                throw failure(400, "请为 LoRA 选择兼容模型。")
            }
            let libraryRoot = LibraryStore.defaultRoot
            let stored = try await Task.detached(priority: .utility) {
                let library = try LibraryStore(root: libraryRoot)
                let directory = try library.managedDirectory("web-adapters").appendingPathComponent(UUID().uuidString)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
                                                        attributes: [.posixPermissions: 0o700])
                let url = directory.appendingPathComponent((name as NSString).deletingPathExtension).appendingPathExtension("safetensors")
                do {
                    try FileManager.default.copyItem(at: source, to: url)
                    return try library.registerLoRA(modelID: model, path: url)
                } catch {
                    try? FileManager.default.removeItem(at: directory)
                    throw error
                }
            }.value
            library.refresh(studio: studio)
            return loraInfo(stored)
        default: throw failure(400, "请选择 .tcapp 应用或 .safetensors LoRA。")
        }
    }

    private func host(_ request: HostRequest) async throws -> Any {
        switch request.action {
        case "download", "planDownload":
            guard !library.busy, let model = request.model, let variant = request.variant,
                  let recipe = LibraryRecipe.all.first(where: { $0.modelID == model }),
                  let selected = recipe.downloadVariants.first(where: { $0.id == variant }) else {
                throw failure(400, "下载正在进行，或所选模型版本不存在。")
            }
            let download = try selected.request(modelID: model, provider: request.provider)
            if request.action == "planDownload" {
                let data = try await LibraryTool.run(["plan", "{request}", "--root", library.root], payload: JSONEncoder().encode(download))
                let plan = try LibraryTool.decode(LibraryDownloadPlan.self, from: data)
                return ["bytes": plan.downloadBytes, "cached": plan.cachedBytes, "files": plan.files.count]
            }
            library.download(download, studio: studio)
        case "cancelDownload": library.cancel()
        case "refresh": library.refresh(studio: studio)
        case "unregisterModel":
            let item = try installIndex().0.installations.first { $0.id == request.id }
            guard !library.busy, let item else { throw failure(404, "找不到模型登记。") }
            library.remove(item, studio: studio)
        case "unregisterLoRA":
            let item = try installIndex().0.loras?.first { $0.id == request.id }
            guard !library.busy, let item else { throw failure(404, "找不到 LoRA 登记。") }
            library.removeLoRA(item)
        case "selectModel":
            guard let selected = try installIndex().1.first(where: { $0.id == request.id }) else { throw failure(404, "找不到模型安装。") }
            if let variant = UpscaleVariant.from(modelID: selected.model) {
                studio.draft.upscaleModelPaths[variant.rawValue] = selected.path
                studio.selectUpscaleVariant(variant)
                studio.save()
            } else { studio.selectInstallation(modelID: selected.model, path: selected.path) }
        case "tinyVAE":
            guard !library.busy, let asset = TinyVAEAsset.all.first(where: { $0.id == request.id }) else {
                throw failure(400, "所选 Tiny VAE 不可用。")
            }
            library.downloadTinyVAE(asset, source: request.variant == "mirror" ? .mirror : .huggingface)
        case "pruna":
            guard !library.busy else { throw failure(409, "模型库有操作正在进行。") }
            library.downloadPrunaVAE(studio: studio)
        case "compileANE", "removeANE":
            guard !library.busy, !store.busy, !store.hasQueuedGeneration,
                  let item = try installIndex().0.anePartitions?.first(where: { $0.id == request.id }) else {
                throw failure(409, "请选取已登记的分区并等待当前任务结束。")
            }
            if request.action == "compileANE" { library.compileANE(item, store: store) }
            else { library.removeANE(item, studio: studio) }
        case "uninstallApp":
            guard let id = request.id, let app = runtime.package(id), !app.builtIn else { throw failure(404, "找不到社区应用。") }
            try runtime.uninstall(app)
        case "pickModel", "pickANE", "pickLoRA", "pickModelRoot", "pickConfiguration", "pickDiagnosticFolder":
            try await pickHostFile(request)
        case "unload": try await store.unload()
        case "preload":
            guard !store.busy, !store.hasQueuedGeneration,
                  let selected = try installIndex().1.first(where: { $0.id == request.id }) else {
                throw failure(409, "请选取可用安装并等待任务结束。")
            }
            let url = URL(fileURLWithPath: selected.path)
            if UpscaleVariant.from(modelID: selected.model) != nil {
                try await store.preloadUpscaler(modelURL: url, compute: .gpu)
            } else if let model = studio.models.first(where: { $0.id == selected.model }) {
                let draft = modelDraft(model, installation: selected)
                let output = outputURL(UUID(), kind: model.isVideo ? "mp4" : "png")
                let native = try await Task.detached { try draft.request(output: output) }.value
                try await store.prepare(modelURL: url, request: native, warmup: false)
            }
        case "releaseUpscaler": await store.releaseUpscaler()
        case "cacheRefresh": await cache.refresh()
        case "cacheRetention", "cachePrune":
            guard let days = request.days, [0, 7, 14, 30, 60, 90].contains(days) else { throw failure(400, "缓存保留天数无效。") }
            if request.action == "cacheRetention" { await cache.retain(days) }
            else {
                guard !cache.busy, !store.busy, !store.hasQueuedGeneration else { throw failure(409, "请等待推理队列空闲。") }
                await cache.prune(store: store, days: days)
            }
        case "clearRuntimeCache": try await store.clearRuntimeProgramCache()
        case "startAPI":
            guard let api else { throw failure(404, "此主机未提供本地 HTTP API 控制。") }
            await api.startShared(store: store)
            guard api.running else { throw failure(409, api.error ?? api.status) }
        case "stopAPI": api?.stop()
        default: throw failure(404, "找不到主机操作。")
        }
        return true
    }

    private func pickHostFile(_ request: HostRequest) async throws {
        guard !hostPickerOpen, !library.busy, let app = NSApp, let window = app.mainWindow ?? app.keyWindow else {
            throw failure(409, "请打开 Mac 上的 TurboCider 窗口，并等待模型库操作结束。")
        }
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        let model = request.model ?? studio.draft.modelID
        switch request.action {
        case "pickModel":
            guard studio.models.contains(where: { $0.id == model && $0.executor }) || UpscaleVariant.from(modelID: model) != nil else {
                throw failure(400, "请先选取一个已知模型。")
            }
            panel.message = "网页管理设备请求登记 \(model) 的本地模型。"
            panel.canChooseDirectories = true
            panel.canChooseFiles = UpscaleVariant.from(modelID: model) != nil
        case "pickModelRoot", "pickDiagnosticFolder":
            panel.message = request.action == "pickModelRoot" ? "选择 TurboCider 的模型库目录。" : "选择要登记的张量输出目录。"
            panel.canChooseDirectories = true; panel.canChooseFiles = false
        default:
            panel.message = "网页管理设备请求导入本地配置。"
            panel.canChooseDirectories = false; panel.canChooseFiles = true; panel.allowedContentTypes = [.json]
        }
        hostPickerOpen = true
        defer { hostPickerOpen = false }
        app.activate(ignoringOtherApps: true)
        let accepted = await withCheckedContinuation { continuation in
            panel.beginSheetModal(for: window) { result in continuation.resume(returning: result == .OK) }
        }
        guard accepted, let file = panel.url else { return }
        try Task.checkCancellation()
        switch request.action {
        case "pickModel":
            if let variant = UpscaleVariant.from(modelID: model) { library.registerUpscaler(file, variant: variant, studio: studio) }
            else {
                let data = try await LibraryTool.run(["register", model, file.path, "--root", library.root])
                let install = try LibraryTool.decode(LibraryInstallation.self, from: data)
                studio.applyLibraryInstallation(install, replace: false)
                library.refresh(studio: studio)
            }
        case "pickANE": library.registerANE(file, modelID: model)
        case "pickLoRA": library.importLoRAManifest(file)
        case "pickModelRoot": library.configure(root: file, studio: studio)
        case "pickConfiguration": library.importPaths(file, studio: studio)
        case "pickDiagnosticFolder": await cache.registerDumps(file)
        default: break
        }
    }
}
