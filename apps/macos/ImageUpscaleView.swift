import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ImageUpscaleView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    var showSettings = true
    @Binding var sourcePath: String
    var manageModels: () -> Void
    var onResult: (NativeJob) -> Void
    @State private var submitting = false
    @State private var resultID: UUID?
    private var locked: Bool { store.busy || submitting || store.externalServiceActive }
    private var result: NativeJob? {
        let candidate = store.jobs.first { $0.id == resultID && $0.hasOutput }
            ?? store.jobs.first { $0.request.operation == "image.upscale" && $0.hasOutput }
        // A newly selected source must not be obscured by an unrelated past result.
        guard sourcePath.isEmpty || candidate?.request.inputs?.first?.path == sourcePath else { return nil }
        return candidate
    }
    var body: some View {
        HStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("图像超分").font(.title2)
                    Text("x2plus 放大 2 倍，x4plus 放大 4 倍。原图保留，结果另存为 PNG。")
                        .foregroundStyle(.secondary)
                    HStack {
                        TextField("要超分的图片路径", text: $sourcePath).textFieldStyle(.roundedBorder)
                            .accessibilityIdentifier("upscaleInputPath")
                        Button("选择图片…") {
                            let panel = NSOpenPanel(); panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = false
                            present(panel) { sourcePath = $0.path }
                        }
                        Button("开始超分 ×\(studio.draft.upscaleVariant.scale)", action: run)
                            .buttonStyle(.borderedProminent).disabled(sourcePath.isEmpty || studio.draft.upscaleModelPath.isEmpty)
                            .accessibilityIdentifier("upscaleStart")
                    }.disabled(locked)
                    if store.busy {
                        HStack {
                            ProgressView().controlSize(.small)
                            if let job = store.activeJob { Text("超分分块 \(job.completed)/\(job.total)") }
                            else { Text(store.sessionState) }
                            Spacer(); Button("取消") { store.cancel() }
                        }
                    }
                    if let message = studio.message { Text(message).foregroundStyle(.orange).textSelection(.enabled) }
                    if let result {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("超分结果 · \(result.request.width) × \(result.request.height)").font(.headline)
                            MediaPreview(path: result.request.output, maxPixel: 1400).frame(height: 360)
                                .accessibilityIdentifier("upscaleOutput")
                            HStack {
                                Text(result.routeSummary ?? "").font(.caption).foregroundStyle(.secondary)
                                Spacer()
                                Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.request.output)]) }
                            }
                        }
                    } else if !sourcePath.isEmpty {
                        Text("待超分原图").font(.headline)
                        MediaPreview(path: sourcePath, maxPixel: 1000).frame(height: 280)
                    } else {
                        ContentUnavailableView("选择要放大的图片", systemImage: "photo.badge.plus",
                                               description: Text("选择原图，打开「超分设置」选择模型后开始超分。"))
                            .frame(height: 280)
                    }
                }.padding(24)
            }
            if showSettings {
                Divider()
                ScrollView {
                    UpscaleSettingsView(store: store, studio: studio, locked: locked, manageModels: manageModels)
                        .padding(18)
                }.frame(width: 290).background(Color(nsColor: .controlBackgroundColor))
            }
        }
        .onDisappear { studio.save() }
    }
    private func run() {
        guard !locked else { return }
        let source = URL(fileURLWithPath: sourcePath), model = URL(fileURLWithPath: studio.draft.upscaleModelPath)
        let compute = studio.draft.upscaleCompute
        submitting = true; studio.message = nil
        Task {
            defer { submitting = false }
            do {
                let job = try await store.upscale(source: source, modelURL: model, compute: compute)
                if let info = store.upscaleReady { studio.rememberUpscaleModel(info) }
                resultID = job.id; onResult(job)
            } catch { studio.message = error is CancellationError ? "超分已取消，原图已保留。" : error.localizedDescription }
        }
    }
    private func present(_ panel: NSOpenPanel, selected: @escaping (URL) -> Void) {
        Task { @MainActor in
            let completion: (NSApplication.ModalResponse) -> Void = { response in
                if response == .OK, let url = panel.url { selected(url) }
            }
            if let window = NSApp.windows.first(where: { $0.isVisible && !($0 is NSPanel) }) {
                panel.beginSheetModal(for: window, completionHandler: completion)
            } else { panel.begin(completionHandler: completion) }
        }
    }
}

/// Shared settings keep generation and standalone upscaling in sync.
struct UpscaleSettingsView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    var locked: Bool
    var showsVariantPicker = true
    var manageModels: () -> Void
    @State private var showDetails = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("超分设置").font(.headline)
            VStack(alignment: .leading, spacing: 10) {
                if showsVariantPicker {
                    Picker("超分模型", selection: Binding(get: { studio.draft.upscaleVariant }, set: { studio.selectUpscaleVariant($0) })) {
                        ForEach(UpscaleVariant.allCases, id: \.self) { Text("\($0.rawValue) · \($0.scale)×").tag($0) }
                    }.accessibilityIdentifier("upscaleVariant")
                } else {
                    Text("\(studio.draft.upscaleVariant.rawValue) · \(studio.draft.upscaleVariant.scale)× 模型")
                        .font(.subheadline)
                    Text("是否超分和放大倍率，在生成按钮旁选择。")
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Text(studio.draft.upscaleModelPath.isEmpty ? "尚未选择模型" : URL(fileURLWithPath: studio.draft.upscaleModelPath).lastPathComponent)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(2).help(studio.draft.upscaleModelPath)
                    Spacer()
                    Button("管理模型", action: manageModels).accessibilityIdentifier("manageUpscaleModels")
                }
                Picker("超分设备", selection: $studio.draft.upscaleCompute) {
                    ForEach(UpscaleCompute.allCases, id: \.self) { Text($0.title).tag($0) }
                }.accessibilityIdentifier("upscaleCompute")
                Toggle("自动预加载", isOn: $studio.draft.upscaleAutoPreload)
                    .accessibilityIdentifier("upscaleAutoPreload")
                DisclosureGroup("模型路径", isExpanded: $showDetails) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(studio.draft.upscaleModelPath.isEmpty ? "请在模型中心下载或导入模型。" : studio.draft.upscaleModelPath)
                            .font(.caption).textSelection(.enabled).accessibilityIdentifier("upscaleModelPath")
                    }.padding(.top, 8)
                }
            }.disabled(locked)
            HStack {
                Button("预加载") { Task { await preload() } }
                    .disabled(locked || studio.draft.upscaleModelPath.isEmpty).accessibilityIdentifier("preloadUpscaler")
                Button("释放") { Task { await store.releaseUpscaler() } }
                    .disabled(locked || store.upscaleReady == nil).accessibilityIdentifier("releaseUpscaler")
            }
            Text(store.upscaleStatus).font(.caption).foregroundStyle(.secondary)
                .accessibilityIdentifier("upscaleStatus")
            Text("原图保留，超分结果另存。预加载后连续复用一个模型；切换模型或设备时重新加载。")
                .font(.caption).foregroundStyle(.secondary)
            if studio.draft.upscaleCompute == .ane {
                Text("ANE 首次准备较慢；预加载后通常更快。不支持的算子可能使用 CPU。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if store.busy && store.activeJob == nil {
                HStack { ProgressView().controlSize(.small); Button("取消预加载") { store.cancel() } }
            }
        }
    }
    private func preload() async {
        studio.message = nil
        do {
            try await store.preloadUpscaler(modelURL: URL(fileURLWithPath: studio.draft.upscaleModelPath), compute: studio.draft.upscaleCompute)
            if let info = store.upscaleReady { studio.rememberUpscaleModel(info) }
        } catch { if !(error is CancellationError) { studio.message = error.localizedDescription } }
    }
}
