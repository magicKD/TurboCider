import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ImageUpscaleView: View {
    @ObservedObject var store: NativeJobStore
    @ObservedObject var studio: StudioState
    var onResult: (NativeJob) -> Void
    @State private var sourcePath = ""
    @State private var submitting = false
    @State private var resultID: UUID?
    private var locked: Bool { store.busy || submitting || store.externalServiceActive }
    private var result: NativeJob? {
        store.jobs.first { $0.id == resultID && $0.hasOutput }
            ?? store.jobs.first { $0.request.operation == "image.upscale" && $0.hasOutput }
    }
    private var preloadKey: String {
        "\(studio.draft.upscaleAutoPreload):\(studio.draft.upscaleModelPath):\(studio.draft.upscaleCompute.rawValue)"
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("图像超分").font(.largeTitle)
                Text("x2plus 放大 2 倍，x4plus 放大 4 倍。原图保留，结果另存为 PNG。")
                    .foregroundStyle(.secondary)
                GroupBox("模型与加速") {
                    VStack(alignment: .leading, spacing: 12) {
                        Picker("超分模型", selection: Binding(get: { studio.draft.upscaleVariant }, set: { studio.selectUpscaleVariant($0) })) {
                            ForEach(UpscaleVariant.allCases, id: \.self) { Text($0.title).tag($0) }
                        }.accessibilityIdentifier("upscaleVariant")
                        HStack {
                            TextField("本地 Core ML 模型路径", text: $studio.draft.upscaleModelPath)
                                .textFieldStyle(.roundedBorder).accessibilityIdentifier("upscaleModelPath")
                            Button("选择模型…") { chooseModel() }
                        }
                        HStack {
                            Link("下载 \(studio.draft.upscaleVariant.rawValue) 预转换模型", destination: studio.draft.upscaleVariant.downloadURL)
                            Link("模型转换与处理说明", destination: UpscaleVariant.processingURL)
                        }
                        Text("下载后解压并选择 .mlpackage；App 原生编译并加载，不需要 Python。模型保留在原目录。")
                            .font(.caption).foregroundStyle(.secondary)
                        Picker("超分设备", selection: $studio.draft.upscaleCompute) {
                            ForEach(UpscaleCompute.allCases, id: \.self) { Text($0.title).tag($0) }
                        }.accessibilityIdentifier("upscaleCompute")
                        Toggle("自动预加载并预热模型", isOn: $studio.draft.upscaleAutoPreload)
                            .accessibilityIdentifier("upscaleAutoPreload")
                    }.disabled(locked)
                    HStack {
                        Button("预加载") { Task { await preload() } }
                            .disabled(locked || studio.draft.upscaleModelPath.isEmpty).accessibilityIdentifier("preloadUpscaler")
                        Button("释放超分模型") { Task { await store.releaseUpscaler() } }
                            .disabled(locked || store.upscaleReady == nil).accessibilityIdentifier("releaseUpscaler")
                        if store.busy && store.activeJob == nil { ProgressView().controlSize(.small) }
                        Text(store.upscaleStatus).font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
                    Text("只保留一个超分模型；连续任务复用它。切换模型或设备时重新加载。ANE 选项允许 CPU 回退，不代表所有算子均在 ANE 上运行。")
                        .font(.caption).foregroundStyle(.secondary).frame(maxWidth: .infinity, alignment: .leading).padding(.top, 4)
                }
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
                    MediaPreview(path: sourcePath, maxPixel: 1000).frame(height: 280)
                }
            }.padding(24)
        }
        .task(id: preloadKey) {
            guard studio.draft.upscaleAutoPreload else { return }
            do { try await Task.sleep(for: .milliseconds(350)) } catch { return }
            guard !locked, (try? ImageUpscaler.validateModelURL(URL(fileURLWithPath: studio.draft.upscaleModelPath))) != nil else { return }
            await preload()
        }
        .onDisappear { studio.save() }
    }
    private func preload() async {
        studio.message = nil
        do {
            try await store.preloadUpscaler(modelURL: URL(fileURLWithPath: studio.draft.upscaleModelPath), compute: studio.draft.upscaleCompute)
            if let info = store.upscaleReady { studio.rememberUpscaleModel(info) }
        } catch { if !(error is CancellationError) { studio.message = error.localizedDescription } }
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
    private func chooseModel() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = true; panel.allowsMultipleSelection = false
        panel.message = "选择解压后的 x2plus 或 x4plus Core ML 模型。"
        present(panel) { url in
            do { try ImageUpscaler.validateModelURL(url); studio.draft.upscaleModelPath = url.path; studio.save() }
            catch { studio.message = error.localizedDescription }
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
