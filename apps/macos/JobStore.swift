import Foundation
import Combine

struct NativeJob: Codable, Identifiable, Sendable {
    let id: UUID
    let createdAt: Date
    let request: NativeRequest
    var state: String
    var phase: String
    var completed: Int
    var total: Int
    var elapsed: Double
    var error: String?
    var resultJSON: String?
    var secondsPerStep: Double?
    var modelPath: String?
    var outputDeleted: Bool?
    var hasOutput: Bool { state == "succeeded" && outputDeleted != true }
    var routeSummary: String? {
        guard let resultJSON, let data = resultJSON.data(using: .utf8),
              let result = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let plan = result["plan"] as? [String: Any], let execution = plan["execution"] as? String else { return nil }
        if execution.hasPrefix("gpu_ane") { return "GPU + Core ML · INT8 · ANE 驻留未知" }
        return result["gpu_graph"] as? String == "compiled_single_blocks" ? "GPU · BF16 · 融合计算块" : "GPU · BF16"
    }
    var isTerminal: Bool { ["succeeded", "failed", "cancelled", "interrupted"].contains(state) }
}

/// Only completed denoise steps count. Repeated boundaries and other phases never
/// become speed samples; transform can start at a non-zero schedule index.
struct StepTelemetry {
    private var lastSequence = -1
    private var boundary: (step: Int, time: Double)?
    private var samples: [Double] = []
    mutating func observe(_ event: NativeEvent) -> Double? {
        guard event.sequence > lastSequence else { return secondsPerStep }
        lastSequence = event.sequence
        guard event.phase == "denoise" else { return secondsPerStep }
        if let previous = boundary, event.completed == previous.step + 1 {
            let duration = event.elapsed_seconds - previous.time
            if duration.isFinite && duration > 0 { samples.append(duration); samples = Array(samples.suffix(5)) }
        } else if let previous = boundary, event.completed < previous.step {
            samples = []
        }
        if boundary == nil || boundary!.step != event.completed { boundary = (event.completed, event.elapsed_seconds) }
        return secondsPerStep
    }
    var secondsPerStep: Double? { samples.count >= 2 ? samples.reduce(0, +) / Double(samples.count) : nil }
}

@MainActor
final class NativeJobStore: ObservableObject {
    @Published private(set) var jobs: [NativeJob] = []
    @Published private(set) var busy = false
    @Published private(set) var storageError: String?
    @Published private(set) var sessionState = "未加载"
    @Published var externalServiceActive = false
    @Published private(set) var loadedPath: String?
    @Published private(set) var loadedModelID: String?
    @Published private(set) var inspectingResources = false
    @Published private(set) var actualRoute: String?
    @Published private(set) var resourceReport: String?
    @Published private(set) var sessionReport: String?
    @Published private(set) var accelerationStatus: String?
    @Published private(set) var resolvingAcceleration = false
    @Published private(set) var deletedJob: NativeJob?
    let directory: URL
    private var engine: NativeEngine?
    private var activeID: UUID?
    private var preparationID: UUID?
    private var cancelRequested = false
    private var tensorCacheTask: Task<Data, Error>?
    private var lastPersist = Date.distantPast
    private var telemetry = StepTelemetry()
    private var lastSequence = -1
    private var denoiseStart: Int?
    private var lastDetailUpdate = 0.0
    var activeJob: NativeJob? { jobs.first { $0.id == activeID } }
    var canUnload: Bool { engine != nil && !busy }

    init(directory: URL) {
        self.directory = directory
        do {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent("outputs"), withIntermediateDirectories: true)
            let file = directory.appendingPathComponent("jobs.json")
            if FileManager.default.fileExists(atPath: file.path) {
                jobs = try JSONDecoder().decode([NativeJob].self, from: Data(contentsOf: file))
                for i in jobs.indices where !jobs[i].isTerminal {
                    jobs[i].state = "interrupted"
                    jobs[i].error = "上次运行已中断，可复用参数重新生成。"
                }
                try persist()
            }
        } catch { storageError = error.localizedDescription }
    }
    private func persist() throws {
        try JSONEncoder().encode(jobs).write(to: directory.appendingPathComponent("jobs.json"), options: .atomic)
        lastPersist = Date()
    }
    func deleteJob(_ id: UUID) throws {
        guard let job = jobs.first(where: { $0.id == id }), job.isTerminal, activeID != id else {
            throw NativeFailure(message: "请先取消或等待任务完成，再删除记录。")
        }
        let previous = jobs
        jobs.removeAll { $0.id == id }
        do { try persist(); deletedJob = job }
        catch { jobs = previous; throw error }
    }
    func undoDeleteJob() throws {
        guard let job = deletedJob else { return }
        let previous = jobs
        jobs.append(job); jobs.sort { $0.createdAt > $1.createdAt }
        do { try persist(); deletedJob = nil }
        catch { jobs = previous; throw error }
    }
    @discardableResult func trashOutput(_ id: UUID) throws -> URL? {
        guard !busy, let index = jobs.firstIndex(where: { $0.id == id }), jobs[index].hasOutput,
              jobs[index].isTerminal, activeID != id else {
            throw NativeFailure(message: "只能删除已完成任务的结果。")
        }
        let url = URL(fileURLWithPath: jobs[index].request.output)
        let canonical = url.resolvingSymlinksInPath()
        let managed = directory.appendingPathComponent("outputs").resolvingSymlinksInPath()
        guard canonical.path.hasPrefix(managed.path + "/"),
              (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw NativeFailure(message: "只能在 App 中删除本 App 输出目录里的结果；外部文件请在 Finder 中管理。")
        }
        guard !(activeJob?.request.inputs ?? []).contains(where: { URL(fileURLWithPath: $0.path).resolvingSymlinksInPath() == canonical }) else {
            throw NativeFailure(message: "当前任务正在使用此图片，请等待完成。")
        }
        var trashed: NSURL?
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.trashItem(at: url, resultingItemURL: &trashed)
        }
        jobs[index].outputDeleted = true
        do { try persist() }
        catch {
            jobs[index].outputDeleted = nil
            if let trashed { try? FileManager.default.moveItem(at: trashed as URL, to: url) }
            throw error
        }
        return trashed as URL?
    }
    /// Resolve on each request so changing an adapter/strength cannot reuse a stale partition.
    func resolveAcceleration(_ draft: StudioDraft) async throws -> StudioDraft {
        guard draft.usesANE, ["flux2-klein-4b", "z-image-turbo"].contains(draft.modelID) else { return draft }
        guard !busy, !resolvingAcceleration else { throw NativeFailure(message: "请等待当前任务完成。") }
        resolvingAcceleration = true
        defer { resolvingAcceleration = false }
        var resolved = draft
        var config = draft.acceleration ?? StudioAcceleration()
        let cache = config.coreMLCache.map { URL(fileURLWithPath: $0) } ?? compilationDirectory
        let preferred = config.manifest, known = config.knownManifests ?? []
        accelerationStatus = "正在检查 ANE 编译缓存…"
        let textTokens = draft.modelID == "z-image-turbo"
            ? try await Task.detached {
                try NativeEngine.zImageTokenCount(modelPath: draft.modelPath, prompt: draft.prompt)
            }.value : 32
        if draft.modelID == "z-image-turbo", textTokens > 1024 {
            throw NativeFailure(message: "Z-Image 提示词为 \(textTokens) tokens，超过当前 1024 上限；GPU 与 ANE 相同，请缩短文本。不会自动截断。")
        }
        let minimumRows = ((draft.width / 16) * (draft.height / 16) + 31) / 32 * 32
            + (textTokens + 31) / 32 * 32
        let match = await Task.detached {
            AccelerationDiscovery.find(modelPath: draft.modelPath, preferred: preferred, cache: cache,
                minimumRows: minimumRows, modelID: draft.modelID, loras: draft.activeLoRAs, knownManifests: known)
        }.value
        try Task.checkCancellation()
        if let match {
            config.manifest = match.manifest
            config.sourceManifest = match.source
            accelerationStatus = "已复用 ANE 编译缓存 · 未重新编译"
        } else {
            let source = config.sourceManifest
            let sourceMatch = await Task.detached {
                let linkedSources = ([preferred] + known).compactMap { file -> String? in
                    guard !file.isEmpty, let data = try? Data(contentsOf: URL(fileURLWithPath: file)),
                          let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
                    return value["source_manifest"] as? String
                }
                let sources = ([source] + linkedSources).filter { !$0.isEmpty }
                return AccelerationDiscovery.find(modelPath: draft.modelPath, preferred: sources.first ?? "", cache: cache,
                    minimumRows: minimumRows, modelID: draft.modelID, loras: draft.activeLoRAs,
                    knownManifests: Array(sources.dropFirst()), requireCompiled: false)
            }.value
            guard let sourceMatch else {
                accelerationStatus = "没有匹配当前模型、LoRA、强度与文本长度的 ANE 缓存"
                throw NativeFailure(message: "没有匹配当前模型、LoRA、强度与文本长度的 ANE 分区（需要 \(minimumRows) 行，文本 \(textTokens) tokens）。请在模型库的“ANE 分区”登记匹配的源 manifest 或编译 manifest。变长分区也有容量上限，且需匹配当前 LoRA 与强度；勾选 ANE 不会自动导出分区。")
            }
            config.sourceManifest = sourceMatch.manifest
            config.coreMLCache = cache.path
            resolved.acceleration = config
            accelerationStatus = "正在检查源分区缓存，仅编译缺失部分…"
            let data = try await coreMLResources(JSONSerialization.data(withJSONObject: resolved.coreMLResourceRequest("compile")))
            guard let report = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let manifest = report["manifest"] as? String else { throw NativeFailure(message: "编译结果缺少分区路径。") }
            config.manifest = manifest
            // Compilation is derived data, but retain its source relationship in
            // the shared inventory so subsequent sessions can discover it.
            _ = try await LibraryTool.run(["register-ane", draft.modelID, manifest])
            let hits = report["cache_hits"] as? Int ?? 0, count = report["partitions"] as? Int ?? 0
            accelerationStatus = "ANE 分区就绪 · 复用 \(hits)/\(count) 个编译缓存"
        }
        config.knownManifests = Array(Set(known + [preferred, config.manifest])).filter { !$0.isEmpty }.sorted()
        resolved.acceleration = config
        return resolved
    }
    private var coreMLResourceBusy = false
    func cancel() {
        guard busy else { return }
        cancelRequested = true
        tensorCacheTask?.cancel()
        if coreMLResourceBusy { NativeEngine.cancelCoreMLResources() }
        engine?.cancel()
        if let id = activeID, let i = jobs.firstIndex(where: { $0.id == id }) {
            jobs[i].state = "cancelling"
            do { try persist() } catch { storageError = error.localizedDescription }
        }
    }
    private func acquire(_ url: URL, modelID: String) async throws -> NativeEngine {
        guard !externalServiceActive else { throw NativeFailure(message: "本地 API 正在运行，请先在 API 页面停止服务。") }
        let path = url.standardizedFileURL.path
        if loadedPath == path, loadedModelID == modelID, let engine { return engine }
        if let old = engine { _ = try await old.unload() }
        engine = nil; loadedPath = nil; loadedModelID = nil
        sessionReport = nil
        sessionState = "正在检查模型…"
        if modelID == "z-image-turbo" {
            let report = await Task.detached(priority: .utility) {
                InstallationInspection.inspect(modelID: modelID, root: url)
            }.value
            guard report.issues.isEmpty else {
                throw NativeFailure(message: "Z-Image 模型检查失败，请在模型库重新选择原始模型目录：\n" + report.issues.map { "\($0.path)：\($0.message)" }.joined(separator: "\n"))
            }
        }
        let opened = try await NativeEngine.open(modelURL: url, modelID: modelID)
        engine = opened; loadedPath = path; loadedModelID = modelID
        sessionState = "会话就绪 · 权重按需加载"
        return opened
    }
    func load(modelURL: URL, modelID: String = "flux2-klein-4b") async throws {
        guard !busy else { throw NativeFailure(message: "任务进行中，请等待完成后加载模型。") }
        busy = true; cancelRequested = false
        defer { busy = false }
        do {
            let opened = try await acquire(modelURL, modelID: modelID)
            if cancelRequested { throw CancellationError() }
            sessionState = "正在加载图像权重…"
            let data = try await opened.load { [weak self] _ in
                Task { @MainActor [weak self] in if self?.cancelRequested == true { self?.engine?.cancel() } }
            }
            resourceReport = String(decoding: data, as: UTF8.self)
            sessionReport = resourceReport
            sessionState = "图像权重已加载 · 文本按需"
        } catch {
            sessionState = engine == nil ? "加载失败" : "会话就绪 · 加载未完成"
            throw error
        }
    }
    func unload() async throws {
        guard !busy else { throw NativeFailure(message: "正在使用模型，暂时无法释放内存。") }
        guard let engine else { return }
        busy = true; sessionState = "正在释放内存…"
        defer { busy = false }
        do {
            let data = try await engine.unload()
            resourceReport = String(decoding: data, as: UTF8.self)
            self.engine = nil; loadedPath = nil; loadedModelID = nil; sessionState = "未加载"
            sessionReport = nil
        } catch { sessionState = "释放失败 · 会话保留"; throw error }
    }
    func pruneTensorCache(days: Int) async throws -> Data {
        guard !busy, !externalServiceActive, !resolvingAcceleration else { throw NativeFailure(message: "请在推理与 API 空闲时清理缓存。") }
        busy = true; cancelRequested = false
        let previous = sessionState; sessionState = "正在清理可重建张量缓存…"
        defer { busy = false; tensorCacheTask = nil; sessionState = previous }
        let work = Task { try await LibraryTool.run(["cache", "prune", String(days)]) }
        tensorCacheTask = work
        return try await work.value
    }
    func coreMLResources(_ payload: Data) async throws -> Data {
        let request = (try JSONSerialization.jsonObject(with: payload)) as? [String: Any]
        if request?["action"] as? String == "inventory" {
            guard !inspectingResources else { throw NativeFailure(message: "正在统计磁盘空间，请稍候。") }
            inspectingResources = true; defer { inspectingResources = false }
            return try await ResourceInventory.run(payload)
        }
        guard !busy else { throw NativeFailure(message: "请等待当前任务完成。") }
        let applying = request?["apply"] as? Bool == true
        guard !externalServiceActive else { throw NativeFailure(message: "请先停止本地 API，再管理加速缓存。") }
        busy = true; coreMLResourceBusy = true; cancelRequested = false
        let id = UUID(); preparationID = id
        defer { busy = false; coreMLResourceBusy = false; preparationID = nil }
        if applying, let engine {
            _ = try await engine.unload(); self.engine = nil; loadedPath = nil; loadedModelID = nil; sessionState = "未加载"
            sessionReport = nil
        }
        let result = try await NativeEngine.coreMLResources(payload) { [weak self] event in
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                if self.cancelRequested { NativeEngine.cancelCoreMLResources() }
                self.preparationEvent(event, id: id)
            }
        }
        resourceReport = String(decoding: result, as: UTF8.self)
        if request?["action"] as? String == "export" { sessionState = "Core ML 源分区已导出" }
        if request?["action"] as? String == "compile" { sessionState = "Core ML 分区已编译" }
        return result
    }
    var compilationDirectory: URL { directory.appendingPathComponent("cache/coreml") }
    private func preparationEvent(_ event: NativeEvent, id: UUID) {
        guard busy, preparationID == id else { return }
        if cancelRequested { engine?.cancel() }
        if event.phase == "transformer_block" || event.phase == "z_image_denoise_block" { return }
        let phase = event.phase == "coreml_compile" ? "预编译" : event.phase == "denoise" ? "预热采样" : event.phase.contains("text") ? "文本准备" : event.phase.contains("coreml") ? "加速分区准备" : "模型准备"
        sessionState = "\(phase) · \(event.completed)/\(event.total) · \(String(format: "%.1f", event.elapsed_seconds)) 秒"
    }
    func prepare(modelURL: URL, request: NativeRequest, warmup: Bool) async throws {
        guard !busy else { throw NativeFailure(message: "请等待当前任务完成。") }
        let id = UUID(); preparationID = id
        busy = true; cancelRequested = false; sessionState = warmup ? "正在预热…" : "正在加载当前配置…"
        defer { preparationID = nil; busy = false }
        do {
            let opened = try await acquire(modelURL, modelID: request.model)
            if cancelRequested { throw CancellationError() }
            let data = try await opened.prepare(request, warmup: warmup) { [weak self] event in
                DispatchQueue.main.async { [weak self] in self?.preparationEvent(event, id: id) }
            }
            resourceReport = String(decoding: data, as: UTF8.self)
            let report = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            sessionReport = resourceReport
            let plan = report?["plan"] as? [String: Any]
            let mode = (report?["execution"] as? String) ?? (plan?["execution"] as? String) ?? "gpu"
            sessionState = (warmup ? "当前任务已预热 · 未保存图片" : "权重与当前文本已就绪") + (mode.hasPrefix("gpu_ane") ? " · GPU + ANE" : " · GPU")
        } catch { sessionState = "准备未完成 · 可重试"; throw error }
    }
    func maintainCache(modelURL: URL, action: String, source: URL? = nil, directory: URL? = nil) async throws -> Data {
        guard !busy else { throw NativeFailure(message: "请等待当前任务完成。") }
        let id = UUID(); preparationID = id
        busy = true; cancelRequested = false; sessionState = "正在管理加速缓存…"
        defer { preparationID = nil; busy = false }
        do {
            let opened = try await acquire(modelURL, modelID: "flux2-klein-4b")
            if cancelRequested { throw CancellationError() }
            let data = try await opened.cache(action: action, directory: directory ?? compilationDirectory, source: source) { [weak self] event in
                DispatchQueue.main.async { [weak self] in self?.preparationEvent(event, id: id) }
            }
            resourceReport = String(decoding: data, as: UTF8.self)
            if action == "clear" { engine = nil; loadedPath = nil; loadedModelID = nil; sessionReport = nil; sessionState = "模型已卸载 · 编译缓存已清除" }
            else { sessionState = action == "compile_manifest" ? "加速分区预编译完成" : "缓存检查完成" }
            return data
        } catch { sessionState = "缓存操作未完成 · 可重试"; throw error }
    }
    func generate(modelURL: URL, request: NativeRequest) async throws -> NativeJob {
        guard !busy else { throw NativeFailure(message: "一次只能生成一张图或一个视频。") }
        guard storageError == nil else { throw NativeFailure(message: storageError!) }
        // Close the reentrancy window before any async plan/session operation.
        busy = true; cancelRequested = false; actualRoute = nil
        defer { busy = false; activeID = nil }
        do { _ = try await Task.detached { try NativeEngine.plan(request) }.value }
        catch { throw error }
        let id = UUID(); activeID = id; telemetry = StepTelemetry(); lastSequence = -1; denoiseStart = nil; lastDetailUpdate = 0
        jobs.insert(NativeJob(id: id, createdAt: Date(), request: request, state: "preparing", phase: "prepare", completed: 0, total: 1, elapsed: 0, modelPath: modelURL.path), at: 0)
        let start = ContinuousClock.now
        do {
            try persist()
            let opened = try await acquire(modelURL, modelID: request.model)
            if cancelRequested { throw CancellationError() }
            sessionState = "使用中"
            let result = try await opened.generate(request) { [weak self] event in
                DispatchQueue.main.async { [weak self] in self?.receive(event, id: id) }
            }
            guard let i = jobs.firstIndex(where: { $0.id == id }) else { throw NativeFailure(message: "Missing job") }
            jobs[i].state = "succeeded"; jobs[i].phase = "complete"
            jobs[i].resultJSON = String(decoding: result, as: UTF8.self)
            sessionReport = jobs[i].resultJSON
            jobs[i].elapsed = Self.seconds(start.duration(to: .now))
            sessionState = request.residency == "component_staged" ? "会话就绪 · 图像权重已释放" : "会话可复用"
            try persist()
            return jobs[i]
        } catch {
            if let i = jobs.firstIndex(where: { $0.id == id }) {
                jobs[i].state = error is CancellationError ? "cancelled" : "failed"
                jobs[i].elapsed = Self.seconds(start.duration(to: .now))
                jobs[i].error = error.localizedDescription
                do { try persist() } catch { storageError = error.localizedDescription }
            }
            sessionState = engine == nil ? "未加载" : "会话就绪 · 可重试"
            throw error
        }
    }
    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
    private func receive(_ event: NativeEvent, id: UUID) {
        guard activeID == id, event.sequence > lastSequence, let i = jobs.firstIndex(where: { $0.id == id }), !jobs[i].isTerminal else { return }
        lastSequence = event.sequence
        if event.phase.hasPrefix("route_") {
            actualRoute = event.phase == "route_gpu_ane" ? "GPU + Core ML · ANE 驻留未知" : "Metal GPU · BF16"
            return
        }
        if cancelRequested { engine?.cancel() }
        // Block callbacks keep cancellation responsive but must not replace the
        // sampling stage or make speed flash briefly between individual blocks.
        if event.phase == "transformer_block" || event.phase == "z_image_denoise_block" {
            if event.elapsed_seconds - lastDetailUpdate >= 0.25 {
                jobs[i].elapsed = event.elapsed_seconds; lastDetailUpdate = event.elapsed_seconds
            }
            return
        }
        var current = jobs[i]
        current.state = cancelRequested ? "cancelling" : "running"
        if event.phase == "denoise", denoiseStart == nil { denoiseStart = event.completed }
        let offset = event.phase == "denoise" ? (denoiseStart ?? 0) : 0
        current.phase = event.phase; current.completed = event.completed - offset; current.total = event.total - offset
        current.elapsed = event.elapsed_seconds
        current.secondsPerStep = telemetry.observe(event)
        jobs[i] = current
        if Date().timeIntervalSince(lastPersist) > 1 {
            do { try persist() } catch { storageError = error.localizedDescription; cancel() }
        }
    }
}
