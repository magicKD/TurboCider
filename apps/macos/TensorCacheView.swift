import SwiftUI
import AppKit

struct TensorCacheView: View {
    @ObservedObject var store: NativeJobStore
    @StateObject private var cache = TensorCacheController()
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("内存与张量缓存").font(.title2)
            Text(store.externalServiceActive ? "本地 API 正在管理推理会话" : store.sessionState).foregroundStyle(.secondary)
            Button("清理内存缓存并卸载会话") { Task { do { try await store.unload() } catch { cache.message = error.localizedDescription } } }
                .disabled(!store.canUnload || store.externalServiceActive)
            Divider()
            Text("文本缓存与输出张量").font(.headline)
            if let report = cache.report {
                Text("\(report.entries.count) 项 · \(ByteCountFormatter.string(fromByteCount: report.bytes, countStyle: .file))").monospacedDigit()
                if report.skipped > 0 { Text("另有 \(report.skipped) 项不属于已识别格式，保留原位。").font(.caption).foregroundStyle(.secondary) }
            }
            Picker("自动保留时间", selection: Binding(get: { cache.retentionDays }, set: { days in Task { await cache.retain(days) } })) {
                Text("不自动清理").tag(0)
                ForEach([7,30,90], id: \.self) { Text("\($0) 天").tag($0) }
                if ![0,7,30,90].contains(cache.retentionDays) { Text("\(cache.retentionDays) 天").tag(cache.retentionDays) }
            }.disabled(cache.busy)
            HStack {
                Button("刷新统计") { Task { await cache.refresh() } }.disabled(cache.busy)
                Button("清理已识别张量缓存") { Task { await cache.prune(store: store, days: 0) } }
                    .disabled(cache.busy || store.busy || store.externalServiceActive || store.resolvingAcceleration || (cache.report?.entries.isEmpty ?? true))
                    .accessibilityIdentifier("pruneTensorCache")
                if cache.busy { ProgressView().controlSize(.small) }
            }
            Text("管理 LTX 文本条件，以及明确登记目录中的 FLUX / Z-Image 输出张量。按生成时间保留，App 运行期间每小时检查一次；推理繁忙时跳过。清理文本缓存后下次请求会重新编码。输出张量需重新生成才能恢复。模型权重、LoRA、ANE 分区、图片视频和未登记的研发输出不会删除。")
                .font(.caption).foregroundStyle(.secondary)
            Button("登记输出张量目录…", action: chooseDumps).disabled(cache.busy)
                .accessibilityIdentifier("registerDiagnosticTensors")
            ForEach(cache.diagnosticDirectories, id: \.self) { path in
                HStack {
                    Text(path).font(.caption).textSelection(.enabled)
                    Spacer()
                    Button("停止管理") { Task { await cache.registerDumps(URL(fileURLWithPath: path), remove: true) } }.disabled(cache.busy)
                }
            }
            if let report = cache.report {
                DisclosureGroup("检查的缓存目录") { ForEach(report.roots, id: \.self) { Text($0).font(.caption).textSelection(.enabled) } }
            }
            if let message = cache.message { Text(message).font(.caption).textSelection(.enabled) }
        }.task { await cache.refresh() }
    }
    private func chooseDumps() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "选择实际 dump 目录。只管理本层能识别的 FLUX / Z-Image safetensors 输出，不递归。登记后使用上方保留期限；不会删除图片视频或模型权重。"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await cache.registerDumps(url) }
    }
}
