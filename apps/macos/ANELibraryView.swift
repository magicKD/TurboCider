import SwiftUI
import AppKit

struct ANELibraryView: View {
    let modelID: String
    @ObservedObject var library: ModelLibraryController
    @ObservedObject var studio: StudioState
    @ObservedObject var store: NativeJobStore
    private var items: [LibraryANEPartition] { library.anePartitions.filter { $0.modelID == modelID } }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("ANE 分区").font(.headline)
                Spacer()
                Button("登记 manifest…") {
                    let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
                    panel.message = "选择完整的源分区或编译分区 manifest；文件保持原位。"
                    if panel.runModal() == .OK, let url = panel.url { library.registerANE(url, modelID: modelID) }
                }.accessibilityIdentifier("registerANE")
            }
            Text("统一登记固定 / 变长源分区和编译缓存。源分区需要先导出；勾选 ANE 不会把 GPU 权重自动转换为 Core ML。").font(.caption).foregroundStyle(.secondary)
            if items.isEmpty { Text("此模型尚未登记 ANE 分区。").font(.caption) }
            ForEach(items) { item in
                VStack(alignment: .leading, spacing: 6) {
                    Text("\(item.kind == "source" ? "源分区 · 待编译或复用缓存" : "编译分区 · 使用前复核本机兼容性") · \(item.capacity)").font(.callout.weight(.medium))
                    Text("\(item.partitionCount) 个分区 · \(item.loras.isEmpty ? "基础模型，无 LoRA" : "绑定 LoRA：" + item.loras.map { URL(fileURLWithPath: $0.path).lastPathComponent + " × " + String($0.strength) }.joined(separator: "、"))").font(.caption)
                    Text(item.path).font(.caption2).textSelection(.enabled)
                    Text("基础权重：\(item.checkpoint)").font(.caption2).foregroundStyle(.secondary).textSelection(.enabled)
                    if let source = item.sourceManifest { Text("源 manifest：\(source)").font(.caption2).textSelection(.enabled) }
                    HStack {
                        Button("使用此分区") { library.useANE(item, studio: studio) }
                        if item.kind == "source" { Button("编译并登记") { library.compileANE(item, store: store) } }
                        Button("检查 / 刷新登记") { library.registerANE(URL(fileURLWithPath: item.path), modelID: modelID) }
                        Button("移除登记") { library.removeANE(item, studio: studio) }
                    }
                }.padding(10).background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 8))
            }
        }.disabled(library.busy || store.busy || store.externalServiceActive)
    }
}
