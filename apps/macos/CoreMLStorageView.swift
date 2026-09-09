import SwiftUI
import AppKit

struct CoreMLStorageView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    @State private var inventory: [StorageEntry] = []
    @State private var reportedStorage: String?
    @State private var total = "尚未统计"
    @State private var pending: Cleanup?
    private struct Cleanup: Identifiable { let id = UUID(); let payload: Data; let paths: String; let bytes: String }
    private struct StorageEntry: Identifiable { var id: String { path }; let path: String; let category: String; let size: String; let error: String? }
    private var config: StudioAcceleration { studio.draft.acceleration ?? StudioAcceleration() }
    private func update(_ change: (inout StudioAcceleration) -> Void) { var c = config; change(&c); c.automaticVersion = 1; studio.draft.acceleration = c }
    private func size(_ value: Any?) -> String { ByteCountFormatter.string(fromByteCount: (value as? NSNumber)?.int64Value ?? 0, countStyle: .file) }
    private var isZImage: Bool { studio.draft.modelID == "z-image-turbo" }
    private var defaultStorage: String {
        reportedStorage ?? store.directory.appendingPathComponent(isZImage ? "coreml/z-image-turbo" : "coreml/flux2-klein-4b/m1088").path
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Core ML 模型与磁盘空间").font(.title2)
            Text("当前分区、编译目录及本 App 的设备专用缓存合计：\(total)").font(.headline)
            HStack {
                Button("刷新占用") { perform("inventory") }.disabled(store.inspectingResources).accessibilityIdentifier("refreshCoreMLStorage")
                Button("清理 App 设备缓存…") { perform("clear_runtime") }
                Button("清理托管编译缓存…") { perform("clear_compiled") }
            }.disabled(store.busy)
            Text("Core ML 设备专用缓存由系统选择位置。这里只清理 TurboCider 的缓存，不清理其他 App 或系统全局缓存；磁盘共享块可能使实际释放空间小于文件大小。").font(.caption).foregroundStyle(.secondary)
            DisclosureGroup("占用明细与所在目录（\(inventory.count) 项）") {
                ForEach(inventory) { entry in
                    HStack(alignment: .top) {
                        VStack(alignment: .leading) {
                            Text("\(entry.category) · \(entry.error ?? entry.size)")
                            Text(entry.path).font(.caption).textSelection(.enabled)
                        }
                        Spacer()
                        Button("定位") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.path)]) }
                    }.padding(.vertical, 3)
                }
            }
            pathRow("当前编译分区", path: config.manifest)
            pathRow("当前源分区", path: config.sourceManifest)
            HStack {
                Button("删除当前编译分区…") { perform("delete_artifacts", kind: "compiled") }.disabled(config.manifest.isEmpty)
                Button("删除当前 Core ML 源模型…") { perform("delete_artifacts", kind: "source") }.disabled(config.sourceManifest.isEmpty)
            }.disabled(store.busy)
            Divider()
            Text("导入并编译 Core ML 分区").font(.headline)
            Text("请选择离线导出工具生成的 manifest。编译与推理均在 App 内原生执行。")
                .font(.caption).foregroundStyle(.secondary)
            Button("选择源分区 manifest…") { chooseFile { path in update { $0.sourceManifest = path } } }
                .disabled(store.busy)
            HStack {
                Button("选择源分区目录…") { chooseDirectory { path in update { $0.coreMLStorage = path } } }
                Button("选择编译缓存目录…") { chooseDirectory { path in update { $0.coreMLCache = path } } }
            }.disabled(store.busy)
            pathRow("源模型输出", path: config.coreMLStorage ?? "使用构建配置；未配置时：\(defaultStorage)")
            pathRow("编译缓存", path: config.coreMLCache ?? "使用构建配置；未配置时：\(store.compilationDirectory.path)")
            HStack {
                Text("Core ML 导出请使用 tools/coreml/export_*.py（开发/发布工具）；正式 App 只导入并编译已导出的分区。")
                    .font(.caption).foregroundStyle(.secondary)
                Button("预编译当前源分区") { perform("compile") }.disabled(config.sourceManifest.isEmpty)
            }.disabled(store.busy)
        }
        .overlay(alignment: .topTrailing) { if store.inspectingResources { ProgressView().controlSize(.small).help("正在统计磁盘空间，生成仍可用") } }
        .task(id: config.manifest + config.sourceManifest + (config.coreMLCache ?? "")) {
            while store.busy { do { try await Task.sleep(for: .milliseconds(100)) } catch { return } }
            if !Task.isCancelled { perform("inventory") }
        }
        .sheet(item: $pending) { cleanup in
            VStack(alignment: .leading, spacing: 14) {
                Text("删除预览 · \(cleanup.bytes)").font(.title2)
                Text("将先卸载当前会话，再永久删除以下项目。外部分区也可能被其他工具使用；原始 safetensors 不在删除范围内。")
                ScrollView { Text(cleanup.paths).font(.caption.monospaced()).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                HStack { Button("取消") { pending = nil }; Spacer(); Button("删除列出的项目", role: .destructive) { pending = nil; apply(cleanup.payload) } }
            }.padding(24).frame(width: 680, height: 440)
        }
    }
    private func pathRow(_ name: String, path: String) -> some View {
        VStack(alignment: .leading, spacing: 2) { Text(name).font(.caption).foregroundStyle(.secondary); Text(path.isEmpty ? "未选择" : path).font(.caption).textSelection(.enabled) }
    }
    private func chooseFile(json: Bool = true, action: (String) -> Void) {
        let panel = NSOpenPanel(); if json { panel.allowedContentTypes = [.json] }
        if panel.runModal() == .OK, let path = panel.url?.path { action(path) }
    }
    private func chooseDirectory(action: (String) -> Void) {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
        if panel.runModal() == .OK, let path = panel.url?.path { action(path) }
    }
    private func request(_ action: String, kind: String?) -> [String: Any] {
        studio.draft.coreMLResourceRequest(action, kind: kind)
    }
    private func perform(_ action: String, kind: String? = nil) {
        let value = request(action, kind: kind)
        Task {
            if action == "inventory" && store.busy { return }
            do {
                let result = try await store.coreMLResources(JSONSerialization.data(withJSONObject: value))
                guard let report = try JSONSerialization.jsonObject(with: result) as? [String: Any] else { return }
                if action == "inventory" {
                    reportedStorage = report["storage"] as? String
                    total = size(report["bytes"])
                    let labels = ["source":"Core ML 源模型", "compiled":"当前编译分区", "managed_compiled":"托管编译目录", "runtime_specialization":"App 设备专用缓存"]
                    inventory = (report["entries"] as? [[String: Any]] ?? []).map { StorageEntry(path: $0["path"] as? String ?? "", category: labels[$0["category"] as? String ?? ""] ?? "资源", size: size($0["bytes"]), error: $0["error"] as? String) }
                } else if let token = report["plan_token"] as? String {
                    var next = value; next["apply"] = true; next["plan_token"] = token
                    let paths = (report["entries"] as? [[String: Any]] ?? []).compactMap { $0["path"] as? String }.joined(separator: "\n")
                    pending = Cleanup(payload: try JSONSerialization.data(withJSONObject: next), paths: paths, bytes: size(report["bytes"]))
                } else {
                    if let source = report["source_manifest"] as? String { update { $0.sourceManifest = source } }
                    if let manifest = report["manifest"] as? String { update { $0.knownManifests = Array(Set(($0.knownManifests ?? []) + [$0.manifest, manifest])).filter { !$0.isEmpty }; $0.manifest = manifest } }
                    studio.message = "ANE 分区就绪 · 复用 \(report["cache_hits"] as? Int ?? 0)/\(report["partitions"] as? Int ?? 0) 个缓存分区。"
                    perform("inventory")
                }
            } catch { studio.message = error is CancellationError ? "资源操作已取消" : error.localizedDescription }
        }
    }
    private func apply(_ payload: Data) {
        Task { do {
            _ = try await store.coreMLResources(payload)
            update { c in
                if !FileManager.default.fileExists(atPath: c.manifest) { c.manifest = "" }
                if !FileManager.default.fileExists(atPath: c.sourceManifest) { c.sourceManifest = "" }
            }
            studio.message = "已删除预览中列出的项目。"; perform("inventory")
        } catch { studio.message = error.localizedDescription } }
    }
}
