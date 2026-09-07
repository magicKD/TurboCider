import SwiftUI
import AppKit

struct AccelerationView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    private var config: StudioAcceleration { studio.draft.acceleration ?? StudioAcceleration(policy: studio.draft.profilePath.isEmpty ? "auto" : "profile") }
    @State private var discoveryMessage = "正在检测本机分区…"
    private func update(_ change: (inout StudioAcceleration) -> Void) { var value = config; change(&value); value.automaticVersion = 1; studio.draft.acceleration = value }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("推理加速与准备").font(.title2)
            Picker("计算模式", selection: Binding(get: { config.policy }, set: { mode in update { $0.policy = mode } })) {
                Text("自动 · 按任务匹配").tag("auto")
                Text("GPU · 原始 BF16").tag("gpu")
                Text("GPU + ANE · INT8 MLP 实验路线").tag("gpu_ane")
                if !studio.draft.profilePath.isEmpty { Text("使用设备配置文件").tag("profile") }
            }.disabled(store.busy).accessibilityIdentifier("accelerationMode")
            if config.policy == "auto" {
                Text(discoveryMessage).font(.caption).foregroundStyle(.secondary)
                Text("分区容量包含文本、输出图和参考图 token；兼容不代表更快。512 与 1024 需分别验证，未匹配时使用 GPU。实际 ANE 驻留由系统决定。").font(.caption).foregroundStyle(.secondary)
                Button("重新检测本机加速") { Task { await discover() } }.disabled(store.busy)
            }
            if config.policy == "gpu_ane" {
                Text("GPU 处理 attention，Core ML 处理量化 MLP；结果可能与纯 GPU 略有不同。实际 ANE 驻留由系统决定。固定分区桶必须容纳文本和所有图片 token。").font(.caption).foregroundStyle(.secondary)
                Button(config.manifest.isEmpty ? "选择已编译分区 manifest…" : "更换已编译分区 manifest…") { choose(compiled: true) }
                if !config.manifest.isEmpty { Text(URL(fileURLWithPath: config.manifest).lastPathComponent).font(.caption).textSelection(.enabled) }
            }
            if config.policy == "gpu" {
                Toggle("编译融合单流计算块", isOn: Binding(get: { config.compileGPU ?? false }, set: { value in update { $0.compileGPU = value } })).disabled(store.busy).accessibilityIdentifier("compileGPU")
                Text("保留块间取消；首次使用新形状会编译，预热可提前完成。不同尺寸的收益需分别实测，不代表整张网络一次融合或固定加速比例。").font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("加载当前配置") { prepare(warmup: false) }.accessibilityIdentifier("prepareModel")
                Button("预热当前任务") { prepare(warmup: true) }.accessibilityIdentifier("warmupModel")
                Button("卸载模型") { Task { do { try await store.unload() } catch { studio.message = error.localizedDescription } } }.disabled(!store.canUnload).accessibilityIdentifier("unloadModel")
            }.disabled(store.busy || studio.draft.modelPath.isEmpty)
            Text("预热使用当前提示词、输入图、尺寸和模式，实际运行一次但不保存结果；更换这些条件后可能需要重新准备。").font(.caption).foregroundStyle(.secondary)
            Divider()
            Button("选择已有 Core ML 源分区…") { choose(compiled: false) }.disabled(store.busy)
            CoreMLStorageView(store: store, studio: studio)
        }.task(id: "\(studio.draft.modelPath)|\(studio.draft.width)|\(studio.draft.height)") { await discover() }.padding(20).background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
    }
    private func discover() async {
        let path = studio.draft.modelPath, preferred = config.manifest
        let minimumRows = (studio.draft.width / 16) * (studio.draft.height / 16) + 1
        let selectedCache = config.coreMLCache.map { URL(fileURLWithPath: $0) }
        let result = await Task.detached { AccelerationDiscovery.find(modelPath: path, preferred: preferred, cache: selectedCache, minimumRows: minimumRows) }.value
        guard !Task.isCancelled, studio.draft.modelPath == path else { return }
        let system = (try? JSONSerialization.jsonObject(with: Data(NativeEngine.system().utf8))) as? [String: Any]
        let supported = system?["gpu"] as? String == "Apple M4 Pro" && (system?["physical_memory_bytes"] as? NSNumber)?.uint64Value == 48 * 1024 * 1024 * 1024
        if let result {
            discoveryMessage = supported ? "已发现本机可用分区 · \(result.rows) token；生成时按实际输入复核。" : "已发现本地分区；此机型尚无自动混合策略验证，自动模式使用 GPU。"
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
