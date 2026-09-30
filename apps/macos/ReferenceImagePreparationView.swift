import SwiftUI
import UniformTypeIdentifiers

private extension UTType {
    static let studioReference = UTType(exportedAs: "org.turbocider.studio-reference")
}

struct StudioReferenceDrag: Codable, Transferable {
    let id: UUID
    let workspaceID: UUID
    static var transferRepresentation: some TransferRepresentation {
        CodableRepresentation(contentType: .studioReference)
    }
}

struct ReferenceImagePreparationView: View {
    let asset: StudioAsset
    @ObservedObject var studio: StudioState
    @Environment(\.dismiss) private var dismiss
    @State private var preset: ReferenceImagePreparation
    @State private var applyToAll = false
    @State private var applying = false
    @State private var error: String?

    init(asset: StudioAsset, studio: StudioState) {
        self.asset = asset
        self.studio = studio
        _preset = State(initialValue: asset.preparation ?? .original)
    }

    private var sourceWidth: Int { asset.original?.width ?? asset.width }
    private var sourceHeight: Int { asset.original?.height ?? asset.height }
    private var dimensions: (width: Int, height: Int)? {
        try? preset.dimensions(width: sourceWidth, height: sourceHeight)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("参考图尺寸").font(.title2.weight(.semibold))
            HStack(spacing: 16) {
                MediaPreview(path: asset.original?.path ?? asset.path, maxPixel: 320)
                    .frame(width: 160, height: 150)
                    .background(.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
                VStack(alignment: .leading, spacing: 8) {
                    Text(asset.name).lineLimit(2)
                    Text("原图 \(sourceWidth) × \(sourceHeight)").foregroundStyle(.secondary)
                    if let dimensions {
                        Text("处理后 \(dimensions.width) × \(dimensions.height)").fontWeight(.medium)
                    } else {
                        Text("图片尺寸无效，请重新导入原图。").foregroundStyle(.red)
                    }
                    Text("等比缩小 · 不裁剪 · 不放大").font(.caption).foregroundStyle(.secondary)
                }.font(.callout)
            }
            Picker("处理方式", selection: $preset) {
                ForEach(ReferenceImagePreparation.allCases) { option in
                    Text(option.title).tag(option)
                }
            }.pickerStyle(.radioGroup).disabled(applying).accessibilityIdentifier("referenceSizePreset")
            Text(preset.detail).font(.caption).foregroundStyle(.secondary)
            if studio.draft.assets.count > 1 {
                Toggle("应用到全部 \(studio.draft.assets.count) 张已添加图片", isOn: $applyToAll)
                    .disabled(applying)
                    .accessibilityIdentifier("prepareAllReferences")
                if applyToAll {
                    Text("每张图片分别保持原比例，实际尺寸可能不同。").font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("原始副本保留，可随时恢复“原图”或撤销。输出画布尺寸保持当前设置。")
                .font(.caption).foregroundStyle(.secondary)
            if studio.draft.modelID == "qwen-image-2.1" {
                Text("Qwen 会按模型规则再次编码参考图；缩小文件不等于减少模型内部的编码尺寸。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error { Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled) }
            HStack {
                if applying { ProgressView().controlSize(.small); Text("正在处理…").font(.caption) }
                Spacer()
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction).disabled(applying)
                Button(preset == .original ? "恢复原图" : "应用尺寸") {
                    applying = true; error = nil
                    let ids = applyToAll ? Set(studio.draft.assets.map(\.id)) : [asset.id]
                    let selectedPreset = preset
                    Task {
                        let succeeded = await studio.prepareAssets(ids: ids, preset: selectedPreset)
                        applying = false
                        if succeeded { dismiss() }
                        else { error = studio.message ?? "未能处理参考图，请重试。" }
                    }
                }.keyboardShortcut(.defaultAction)
                    .disabled(applying || studio.importing || dimensions == nil || !studio.draft.assets.contains(where: { $0.id == asset.id }))
                    .accessibilityIdentifier("applyReferenceSize")
            }
        }.padding(24).frame(width: 500)
            .interactiveDismissDisabled(applying)
    }
}
