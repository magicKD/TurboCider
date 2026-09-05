import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers

@main
struct TurboCiderNativeApp: App {
    @StateObject private var store: NativeJobStore
    init() {
        let directory = ProcessInfo.processInfo.environment["TURBOCIDER_NATIVE_STATE"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("TurboCiderNative")
        _store = StateObject(wrappedValue: NativeJobStore(directory: directory))
    }
    var body: some Scene {
        WindowGroup("TurboCider") { StudioView(store: store).frame(minWidth: 1080, minHeight: 740) }
    }
}
private struct StudioModel: Decodable, Identifiable {
    var id: String
    var name: String
    var executor: Bool
    var output: String
    var operations: [String]
    var roles: [String]
    var default_steps: Int
    var default_frames: Int
    var default_width: Int
    var default_height: Int
}
struct StudioView: View {
    @ObservedObject var store: NativeJobStore
    @AppStorage("TurboCiderNativeModelPath") private var modelPath = ""
    @State private var modelID = "flux2-klein-4b"
    @State private var operation = "image.generate"
    @State private var prompt = "A red fox sitting in a snowy forest, soft morning light, detailed photography."
    @State private var width = 512
    @State private var height = 512
    @State private var frames = 22
    @State private var seed = 42
    @State private var steps = 4
    @State private var inputs: [NativeInput] = []
    @State private var profilePath = ""
    @State private var residency = "resident"
    @State private var strength = 0.5
    @State private var dynamicText = true
    @State private var selected: UUID?
    @State private var message: String?
    private var models: [StudioModel] {
        let data = Data(NativeEngine.models().utf8)
        struct Registry: Decodable { var models: [StudioModel] }
        return (try? JSONDecoder().decode(Registry.self, from: data).models) ?? []
    }
    private var model: StudioModel? { models.first { $0.id == modelID } }
    private var job: NativeJob? { store.jobs.first(where: { $0.id == selected }) ?? store.jobs.first }
    private var video: Bool { model?.output == "video" }
    private func operationName(_ value: String) -> String {
        ["image.generate":"文生图", "image.transform":"图生图", "image.edit":"参考图编辑", "video.generate":"文生视频", "video.keyframes":"首尾帧视频", "video.reference":"参考媒体视频", "video.image":"首帧生视频"][value] ?? value
    }
    var body: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 16) {
                Label("TurboCider", systemImage: "sparkles").font(.largeTitle.bold())
                Text("本机多模态创作").foregroundStyle(.secondary)
                Divider()
                List(store.jobs, selection: $selected) { item in
                    VStack(alignment: .leading) {
                        Text(item.request.prompt).lineLimit(2)
                        Text("\(item.state) · \(item.request.width) × \(item.request.height)").font(.caption).foregroundStyle(.secondary)
                    }.tag(item.id)
                }.listStyle(.sidebar)
                Button("系统信息") { message = NativeEngine.system() }
            }.padding().navigationSplitViewColumnWidth(245)
        } detail: {
            HSplitView {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("创作").font(.title.bold())
                        Picker("模型", selection: Binding(get: { modelID }, set: { selectModel($0) })) {
                            ForEach(models) { Text($0.name).tag($0.id) }
                        }.disabled(store.busy)
                        if modelID != "flux2-klein-4b" {
                            Text(model?.executor == true ? "原生计算已接入，真实模型验收待完成" : "模型执行器迁移中，暂不能生成").font(.caption).foregroundStyle(.secondary)
                        }
                        HStack {
                            Text(modelPath.isEmpty ? "选择已有模型文件夹" : URL(fileURLWithPath: modelPath).lastPathComponent).lineLimit(1).font(.caption)
                            Spacer(); Button("选择…", action: chooseModel).disabled(store.busy)
                        }
                        Picker("创作方式", selection: Binding(get: { operation }, set: { operation = $0; inputs = [] })) {
                            ForEach(model?.operations ?? [], id: \.self) { Text(operationName($0)).tag($0) }
                        }.disabled(store.busy)
                        TextEditor(text: $prompt).frame(height: 125).border(.quaternary).accessibilityIdentifier("prompt")
                        if !operation.hasSuffix("generate") {
                            HStack {
                                if operation == "video.keyframes" {
                                    Button("添加首帧") { importAssets(role: "first_frame") }
                                    Button("添加尾帧") { importAssets(role: "last_frame") }
                                } else { Button("添加参考素材…") { importAssets(role: operation == "image.transform" ? "init_image" : operation == "video.image" ? "first_frame" : "reference") } }
                            }.disabled(store.busy)
                            ForEach(inputs) { asset in
                                HStack {
                                    if asset.kind == "image", let preview = NSImage(contentsOfFile: asset.path) {
                                        Image(nsImage: preview).resizable().scaledToFit().frame(width: 40, height: 40)
                                    }
                                    Text(URL(fileURLWithPath: asset.path).lastPathComponent).lineLimit(1).font(.caption)
                                    Spacer()
                                    if inputs.count > 1 {
                                        Button { moveAsset(asset, -1) } label: { Image(systemName: "arrow.up") }.help("向前移动参考图")
                                        Button { moveAsset(asset, 1) } label: { Image(systemName: "arrow.down") }.help("向后移动参考图")
                                    }
                                    Button { inputs.removeAll { $0.id == asset.id } } label: { Image(systemName: "xmark.circle") }
                                }
                            }
                            if operation == "image.transform" {
                                Slider(value: $strength, in: 0...1) { Text("原图强度") }
                                Text(String(format: "原图强度 %.2f", strength)).font(.caption)
                            }
                        }
                        HStack { TextField("宽", value: $width, format: .number); Text("×"); TextField("高", value: $height, format: .number) }
                        if video { TextField("帧数", value: $frames, format: .number) }
                        HStack { Text("种子"); TextField("种子", value: $seed, format: .number); Text("步数"); TextField("步数", value: $steps, format: .number) }
                        DisclosureGroup("性能设置") {
                            VStack(alignment: .leading) {
                                Toggle("按提示词长度优化文本计算", isOn: $dynamicText)
                                Picker("权重驻留", selection: $residency) {
                                    Text("驻留，加快连续生成").tag("resident")
                                    Text("分阶段释放").tag("component_staged")
                                    if modelID == "minimax-h3-turbo" { Text("磁盘流式读取").tag("streamed") }
                                }
                                Button(profilePath.isEmpty ? "选择本机加速配置…" : URL(fileURLWithPath: profilePath).lastPathComponent) { chooseProfile() }
                                if !profilePath.isEmpty { Button("恢复默认 GPU") { profilePath = "" } }
                                Text("默认使用 Metal GPU；加速配置需要与本机匹配，才能开启 GPU / ANE 分工。").font(.caption).foregroundStyle(.secondary)
                            }.padding(.top, 8)
                        }
                        HStack {
                            Button(video ? "生成视频" : "生成图片", action: generate).buttonStyle(.borderedProminent)
                                .disabled(store.busy || model?.executor != true || modelPath.isEmpty || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                                .accessibilityIdentifier("generate")
                            if store.busy { Button("取消", role: .cancel) { store.cancel() } }
                        }
                        if store.busy, let active = store.jobs.first {
                            ProgressView(value: Double(active.completed), total: Double(max(1, active.total)))
                            Text("\(active.phase)  \(active.completed) / \(active.total)").font(.caption.monospaced())
                            Text(String(format: "已用时 %.1f 秒", active.elapsed)).foregroundStyle(.secondary)
                        }
                        if let error = message ?? store.storageError { Text(error).foregroundStyle(.red).textSelection(.enabled).font(.caption) }
                    }.padding(24)
                }.frame(minWidth: 350, idealWidth: 400, maxWidth: 460)
                VStack(spacing: 16) {
                    if let item = job, item.state == "succeeded" {
                        if item.request.output.hasSuffix(".mp4") {
                            VideoPlayer(player: AVPlayer(url: URL(fileURLWithPath: item.request.output)))
                        } else if let image = NSImage(contentsOfFile: item.request.output) {
                            Image(nsImage: image).resizable().scaledToFit().accessibilityIdentifier("generatedImage")
                        }
                        HStack {
                            Text("\(item.request.width) × \(item.request.height) · seed \(item.request.seed)").font(.caption)
                            Button("在 Finder 中显示") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.request.output)]) }
                            Button("另存为…") { exportResult(item.request.output) }
                            Button("复用参数") { reuse(item.request) }.disabled(store.busy)
                        }
                    } else {
                        Image(systemName: "photo.on.rectangle.angled").font(.system(size: 60)).foregroundStyle(.secondary)
                        Text(store.busy ? "正在本机生成…" : "创作结果会显示在这里").foregroundStyle(.secondary)
                        if let error = job?.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                    }
                }.padding(24).frame(maxWidth: .infinity, maxHeight: .infinity).background(.black.opacity(0.04))
            }
        }.onAppear { if modelPath.isEmpty { modelPath = ProcessInfo.processInfo.environment["TURBOCIDER_FLUX_MODEL"] ?? "" } }
    }
    private func selectModel(_ id: String) {
        UserDefaults.standard.set(modelPath, forKey: "modelPath.\(modelID)")
        modelID = id
        if let model { operation = model.operations.first ?? ""; steps = model.default_steps; width = model.default_width; height = model.default_height; frames = model.default_frames }
        inputs = []; residency = "resident"
        modelPath = UserDefaults.standard.string(forKey: "modelPath.\(id)") ?? ""
    }
    private func chooseModel() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { modelPath = url.path; UserDefaults.standard.set(modelPath, forKey: "modelPath.\(modelID)") }
    }
    private func chooseProfile() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url { profilePath = url.path }
    }
    private func moveAsset(_ asset: NativeInput, _ offset: Int) {
        guard let index = inputs.firstIndex(where: { $0.id == asset.id }), inputs.indices.contains(index + offset) else { return }
        inputs.swapAt(index, index + offset)
    }
    private func exportResult(_ path: String) {
        let panel = NSSavePanel(); panel.allowedContentTypes = path.hasSuffix(".mp4") ? [.mpeg4Movie] : [.png]
        panel.nameFieldStringValue = URL(fileURLWithPath: path).lastPathComponent
        if panel.runModal() == .OK, let destination = panel.url {
            do { try Data(contentsOf: URL(fileURLWithPath: path)).write(to: destination, options: .atomic) }
            catch { message = error.localizedDescription }
        }
    }
    private func importAssets(role: String) {
        let panel = NSOpenPanel(); panel.allowsMultipleSelection = role == "reference"
        panel.allowedContentTypes = operation == "video.reference" ? [.image, .movie, .audio] : [.image]
        if panel.runModal() == .OK {
            if role != "reference" { inputs.removeAll { $0.role == role } }
            for url in panel.urls {
                let type = (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType) ?? .image
                let kind = type.conforms(to: .movie) ? "video" : type.conforms(to: .audio) ? "audio" : "image"
                let asset = NativeInput(kind: kind, role: role, path: url.path)
                if !inputs.contains(where: { $0.id == asset.id }) { inputs.append(asset) }
            }
        }
    }
    private func reuse(_ request: NativeRequest) {
        if modelID != request.model { selectModel(request.model) }
        prompt = request.prompt; width = request.width; height = request.height; seed = request.seed; steps = request.steps; frames = request.frames
        operation = request.operation ?? "image.generate"; inputs = request.inputs ?? []; profilePath = request.profile ?? ""; residency = request.residency ?? "resident"
        dynamicText = request.dynamic_text
        strength = request.inputs?.first(where: { $0.role == "init_image" })?.strength ?? 0.5
    }
    private func generate() {
        message = nil
        var request = NativeRequest(prompt: prompt, output: store.directory.appendingPathComponent("outputs/\(UUID().uuidString).\(video ? "mp4" : "png")").path)
        request.model = modelID; request.operation = operation; request.width = width; request.height = height; request.seed = seed; request.steps = steps; request.frames = video ? frames : 1
        request.dynamic_text = dynamicText
        request.residency = residency; request.profile = profilePath.isEmpty ? nil : profilePath
        request.inputs = inputs.map { var asset = $0; if asset.role == "init_image" { asset.strength = strength }; return asset }
        Task { @MainActor in
            do { let item = try await store.generate(modelURL: URL(fileURLWithPath: modelPath), request: request); selected = item.id }
            catch is CancellationError { message = "生成已取消" }
            catch { message = error.localizedDescription }
        }
    }
}
