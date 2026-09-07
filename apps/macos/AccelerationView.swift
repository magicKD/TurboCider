import SwiftUI
import AppKit

struct AccelerationView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    private var model: StudioModel? { studio.models.first { $0.id == studio.draft.modelID } }
    private var supportsGPUANE: Bool { model?.supports_gpu_ane == true }
    private var supportsAutomaticGPUANE: Bool {
        studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo"
    }
    private var config: StudioAcceleration { studio.draft.acceleration ?? StudioAcceleration(policy: studio.draft.profilePath.isEmpty ? "auto" : "profile") }
    @State private var discoveryMessage = "正在检测本机分区…"
    private func update(_ change: (inout StudioAcceleration) -> Void) { var value = config; change(&value); value.automaticVersion = 1; studio.draft.acceleration = value }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("推理加速与准备").font(.title2)
            Picker("计算模式", selection: Binding(get: { config.policy }, set: { mode in update { $0.policy = mode } })) {
                Text(supportsAutomaticGPUANE ? "自动适配本机 · 优先 GPU + ANE" : "自动适配本机 · GPU").tag("auto")
                Text("GPU · 原始 BF16").tag("gpu")
                if supportsGPUANE { Text("GPU + ANE · INT8 MLP 实验路线").tag("gpu_ane") }
                if !studio.draft.profilePath.isEmpty { Text("使用设备配置文件").tag("profile") }
            }.disabled(store.busy).accessibilityIdentifier("accelerationMode")
            if config.policy == "auto" {
                Text(discoveryMessage).font(.caption).foregroundStyle(.secondary)
                Text(supportsAutomaticGPUANE ? "兼容时使用 INT8 混合路线；容量不足或分区加载失败时回退 GPU。实际 ANE 驻留由系统决定。" : "当前模型尚无通过端到端性能门禁的自动 GPU + ANE 配置，自动模式使用稳定路径。")
                    .font(.caption).foregroundStyle(.secondary)
                if supportsAutomaticGPUANE { Button("重新检测本机加速") { Task { await discover() } }.disabled(store.busy) }
            }
            if config.policy == "gpu_ane" {
                Text("GPU 处理 attention，Core ML 处理量化 MLP；结果可能与纯 GPU 略有不同。实际 ANE 驻留由系统决定。固定分区桶必须容纳文本和所有图片 token。").font(.caption).foregroundStyle(.secondary)
                if (studio.draft.modelID.hasPrefix("flux2-") || studio.draft.modelID == "z-image-turbo") && !studio.draft.loras.isEmpty {
                    Text("带 LoRA 的 GPU + ANE 必须选择用同一独立 LoRA 导出的分区；否则 App 会安全切到 GPU。运行时会严格核验路径、大小、SHA-256、角色和强度。")
                        .font(.caption).foregroundStyle(.orange)
                }
                Button(config.manifest.isEmpty ? "选择已编译分区 manifest…" : "更换已编译分区 manifest…") { choose(compiled: true) }
                if !config.manifest.isEmpty { Text(URL(fileURLWithPath: config.manifest).lastPathComponent).font(.caption).textSelection(.enabled) }
            }
            if config.policy == "gpu" {
                if studio.draft.modelID.hasPrefix("flux2-") {
                    Toggle("编译融合单流计算块", isOn: Binding(get: { config.compileGPU ?? false }, set: { value in update { $0.compileGPU = value } })).disabled(store.busy).accessibilityIdentifier("compileGPU")
                    Text("保留块间取消；首次使用新形状会编译，预热可提前完成。当前验证收益约 2%，并非整张网络一次融合。").font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("当前模型使用自身的 native Metal/MLX 图与 pipeline cache；没有可单独开启的 FLUX 单块编译选项。")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack {
                Button("加载当前配置") { prepare(warmup: false) }.accessibilityIdentifier("prepareModel")
                Button("预热当前任务") { prepare(warmup: true) }.accessibilityIdentifier("warmupModel")
                Button("卸载模型") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }.disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
            }.disabled(store.busy || studio.draft.modelPath.isEmpty)
            Text("预热使用当前提示词、输入图、尺寸和模式，实际运行一次但不保存结果；更换这些条件后可能需要重新准备。").font(.caption).foregroundStyle(.secondary)
            Divider()
            if studio.draft.modelID == "flux2-klein-4b" || studio.draft.modelID == "z-image-turbo" {
                Button("选择已有 Core ML 源分区…") { choose(compiled: false) }.disabled(store.busy)
                CoreMLStorageView(store: store, studio: studio)
            }
        }.task(id: studio.draft.modelPath) { await discover() }.padding(20).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
    }
    private func discover() async {
        guard supportsAutomaticGPUANE else {
            discoveryMessage = supportsGPUANE
                ? "当前模型的 GPU + ANE 仍为显式实验配置；自动模式不会启用。"
                : "当前模型使用稳定 GPU 路径。"
            return
        }
        let path = studio.draft.modelPath, preferred = config.manifest
        let selectedCache = config.coreMLCache.map { URL(fileURLWithPath: $0) }
        let modelID = studio.draft.modelID
        let loras = studio.draft.loras
        let result = await Task.detached { AccelerationDiscovery.find(modelPath: path, preferred: preferred, cache: selectedCache, enforceAutomaticPolicy: true, modelID: modelID, loras: loras) }.value
        guard studio.draft.modelPath == path else { return }
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
        if panel.runModal() == .OK, let url = panel.url { update { if compiled { $0.manifest = url.path } else { $0.sourceManifest = url.path } } }
    }
    private func prepare(warmup: Bool) {
        do {
            let request = try studio.draft.request(output: store.directory.appendingPathComponent("unused-warmup.png"))
            let modelURL = URL(fileURLWithPath: studio.draft.modelPath); studio.message = nil
            Task { do { try await store.prepare(modelURL: modelURL, request: request, warmup: warmup) } catch { studio.message = error is CancellationError ? "准备已取消" : error.localizedDescription } }
        } catch { studio.message = error.localizedDescription }
    }
}
