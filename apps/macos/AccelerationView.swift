import SwiftUI
import AppKit

struct AccelerationView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @ObservedObject var library: ModelLibraryController
    private var model: StudioModel? { studio.models.first { $0.id == studio.draft.modelID } }
    private var supportsGPUANE: Bool { model?.supports_gpu_ane == true }
    private var supportsAutomaticGPUANE: Bool {
        studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo"
    }
    private var automaticBucket: Int? {
        AccelerationDiscovery.automaticBucket(
            modelID: studio.draft.modelID, operation: studio.draft.operation,
            width: studio.draft.width, height: studio.draft.height,
            steps: studio.draft.steps, residency: studio.draft.residency,
            hasInputs: !studio.draft.activeAssets.isEmpty)
    }
    private var discoveryID: String {
        let adapters = studio.draft.activeLoRAs.map { "\($0.path):\($0.strength):\($0.role)" }.joined(separator: "|")
        return "\(studio.draft.modelID)|\(studio.draft.modelPath)|\(studio.draft.operation)|\(studio.draft.width)x\(studio.draft.height)|\(studio.draft.steps)|\(studio.draft.residency)|\(adapters)"
    }
    private var config: StudioAcceleration { studio.draft.acceleration ?? StudioAcceleration(policy: studio.draft.profilePath.isEmpty ? "gpu" : "profile") }
    @State private var zImageBucket: Int?
    private var zImageCapacity: String? {
        guard studio.draft.modelID == "z-image-turbo", let zImageBucket else { return nil }
        let imageRows = ((studio.draft.width / 16) * (studio.draft.height / 16) + 31) / 32 * 32
        let textRows = min(512, max(0, (zImageBucket - imageRows) / 32 * 32))
        return textRows == 0
            ? "此分区无法容纳当前尺寸和提示词，请选择更大的分区或使用 GPU。"
            : "当前尺寸可容纳最多 \(textRows) 个编码后的文本 token；更长的提示词需要更大分区或使用 GPU。"
    }
    @State private var discoveryMessage = "正在检测本机分区…"
    private func update(_ change: (inout StudioAcceleration) -> Void) { var value = config; change(&value); value.automaticVersion = 1; studio.draft.acceleration = value }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("推理加速与准备").font(.title2)
            Toggle("GPU", isOn: .constant(true)).toggleStyle(.checkbox).disabled(true)
            Toggle("额外启用 ANE", isOn: Binding(get: { studio.draft.usesANE }, set: { studio.setANEEnabled($0) }))
                .toggleStyle(.checkbox).disabled(store.busy || !supportsGPUANE).accessibilityIdentifier("enableANE")
            Text("默认只使用 GPU；勾选 ANE 后先复用已编译缓存，缺失时才编译源分区。").font(.caption).foregroundStyle(.secondary)
            if studio.draft.usesANE, let status = store.accelerationStatus { Text(status).font(.caption).foregroundStyle(.secondary) }
            if config.policy == "auto" {
                Text(discoveryMessage).font(.caption).foregroundStyle(.secondary)
                Text(supportsAutomaticGPUANE
                     ? "只有机型、任务、尺寸、步数、token 容量和分区几何都命中实测案例时才使用混合路线；其余情况回退 GPU。实际 ANE 驻留由系统决定。"
                     : "当前模型尚无通过端到端性能门禁的自动 GPU + ANE 配置，自动模式使用稳定路径。")
                    .font(.caption).foregroundStyle(.secondary)
                if supportsAutomaticGPUANE { Button("重新检测本机加速") { Task { await discover() } }.disabled(store.busy) }
            }
            if config.policy == "gpu_ane" {
                Text("GPU 处理 attention，Core ML 处理量化 MLP；结果可能与纯 GPU 略有不同。实际 ANE 驻留由系统决定。固定或变长分区的容量必须容纳文本和所有图片 token。").font(.caption).foregroundStyle(.secondary)
                if (studio.draft.modelID.hasPrefix("flux2-") || studio.draft.modelID == "z-image-turbo") && !studio.draft.activeLoRAs.isEmpty {
                    Text("带 LoRA 的 ANE 加速需要匹配同一文件与强度的分区。没有匹配缓存时会提示选择对应分区，或关闭 ANE 使用 GPU。")
                        .font(.caption).foregroundStyle(.orange)
                }
                Button(config.manifest.isEmpty ? "选择已编译分区 manifest…" : "更换已编译分区 manifest…") { choose(compiled: true) }
                if !config.manifest.isEmpty { Text(URL(fileURLWithPath: config.manifest).lastPathComponent).font(.caption).textSelection(.enabled) }
                if let zImageCapacity { Text(zImageCapacity).font(.caption).foregroundStyle(.secondary) }
            }
            if config.policy == "gpu" {
                if studio.draft.modelID.hasPrefix("flux2-") {
                    Toggle("编译融合单流计算块", isOn: Binding(get: { config.compileGPU ?? false }, set: { value in update { $0.compileGPU = value } })).disabled(store.busy).accessibilityIdentifier("compileGPU")
                    Text("保留块间取消；首次使用新形状会编译，预热可提前完成。不同尺寸的收益需分别实测，并非整张网络一次融合。").font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("当前模型使用自身的 native Metal/MLX 图与 pipeline cache；没有可单独开启的 FLUX 单块编译选项。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack {
                Button("加载当前配置") { prepare(warmup: false) }.accessibilityIdentifier("prepareModel")
                Button("预热当前任务") { prepare(warmup: true) }.accessibilityIdentifier("warmupModel")
                Button("卸载模型") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }.disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
            }.disabled(store.busy || store.externalServiceActive || studio.draft.modelPath.isEmpty)
            Text("预热使用当前提示词、输入图、尺寸和模式，实际运行一次但不保存结果；更换这些条件后可能需要重新准备。").font(.caption).foregroundStyle(.secondary)
            Divider()
            if studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo" {
                Button("选择已有 Core ML 源分区…") { choose(compiled: false) }.disabled(store.busy)
                CoreMLStorageView(store: store, studio: studio)
            }
        }.task(id: discoveryID) { await discover() }
            .task(id: "\(studio.draft.modelID)|\(config.manifest)") { await readZImageBucket() }
            .padding(20).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
    }
    private func readZImageBucket() async {
        zImageBucket = nil
        guard studio.draft.modelID == "z-image-turbo", !config.manifest.isEmpty else { return }
        let path = config.manifest
        // File access may wait for macOS permissions; never do it during rendering.
        let bucket = await Task.detached(priority: .utility) { () -> Int? in
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
                  let manifest = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let shape = manifest["shape"] as? [String: Any],
                  let buckets = shape["buckets"] as? [Int], !buckets.isEmpty else { return nil }
            return buckets.max()
        }.value
        guard !Task.isCancelled else { return }
        zImageBucket = bucket
    }
    private func discover() async {
        guard supportsAutomaticGPUANE else {
            discoveryMessage = supportsGPUANE
                ? "当前模型的 GPU + ANE 仍为显式实验配置；自动模式不会启用。"
                : "当前模型使用稳定 GPU 路径。"
            return
        }
        guard let requiredRows = automaticBucket else {
            discoveryMessage = "当前操作、尺寸、步数或驻留模式没有实测混合案例，自动模式使用 GPU。"
            return
        }
        let path = studio.draft.modelPath, preferred = config.manifest
        let minimumRows = (studio.draft.width / 16) * (studio.draft.height / 16) + 1
        let selectedCache = config.coreMLCache.map { URL(fileURLWithPath: $0) }
        let modelID = studio.draft.modelID
        let loras = studio.draft.activeLoRAs
        let result = await Task.detached {
            AccelerationDiscovery.find(modelPath: path, preferred: preferred,
                                       cache: selectedCache, minimumRows: minimumRows,
                                       requiredRows: requiredRows,
                                       enforceAutomaticPolicy: true,
                                       modelID: modelID, loras: loras)
        }.value
        guard !Task.isCancelled, studio.draft.modelPath == path else { return }
        let system = (try? JSONSerialization.jsonObject(with: Data(NativeEngine.system().utf8))) as? [String: Any]
        let gpu = system?["gpu"] as? String
        let memory = (system?["physical_memory_bytes"] as? NSNumber)?.uint64Value
        let supported = (gpu == "Apple M4 Pro" && memory == 48 * 1024 * 1024 * 1024) ||
            (gpu == "Apple M4 Max" && memory == 64 * 1024 * 1024 * 1024)
        if let result {
            discoveryMessage = supported ? "已发现本机可用分区 · \(result.rows) token · ANE MLP \(result.aneMLPEnd - result.aneMLPStart)/\(result.mlpWidth)；生成时按实际输入复核。" : "已发现本地分区；此机型尚无自动混合策略验证，自动模式使用 GPU。"
            if config.policy == "auto" || config.manifest.isEmpty { update { $0.manifest = result.manifest; $0.sourceManifest = result.source } }
        } else { discoveryMessage = "未发现匹配当前权重的完整编译分区，自动模式使用 GPU。可先预编译或指定本地分区。" }
    }
    private func choose(compiled: Bool) {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        panel.message = compiled ? "选择引用 .mlmodelc 的分区 manifest" : "选择引用 .mlpackage 的源分区 manifest"
        if panel.runModal() == .OK, let url = panel.url {
            library.registerANE(url, modelID: studio.draft.modelID, studio: studio, select: true)
        }
    }
    private func prepare(warmup: Bool) {
        guard !store.externalServiceActive else { studio.message = "请先停止本地 API。"; return }
        let snapshot = studio.draft
        studio.message = nil
        Task { do {
            let resolved = try await store.resolveAcceleration(snapshot)
            studio.rememberAcceleration(resolved)
            let output = store.directory.appendingPathComponent("unused-warmup.png")
            let request = try await Task.detached { try resolved.request(output: output) }.value
            try await store.prepare(modelURL: URL(fileURLWithPath: resolved.modelPath), request: request, warmup: warmup)
        } catch { studio.message = error is CancellationError ? "准备已取消" : error.localizedDescription } }
    }
}
