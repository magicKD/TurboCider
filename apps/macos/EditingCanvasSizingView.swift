import SwiftUI

struct EditingCanvasSizingView: View {
    @ObservedObject var studio: StudioState
    let initialAssetID: UUID?
    let submissionLocked: Bool
    @Environment(\.dismiss) private var dismiss
    @State private var assetID: UUID?
    @State private var preset = EditingCanvasPreset.automatic512

    private var selectedID: UUID? {
        assetID ?? initialAssetID ?? studio.draft.activeAssets.first?.id
    }
    private var suggestion: EditingCanvasSuggestion {
        studio.editingCanvasSuggestion(preset: preset, assetID: selectedID)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("根据参考图设置画布").font(.title2.weight(.semibold))
            Text("选择输出图片的尺寸，参考图文件保持不变。")
                .font(.callout).foregroundStyle(.secondary)
            Picker("参考图", selection: Binding(get: { selectedID }, set: { assetID = $0 })) {
                ForEach(Array(studio.draft.activeAssets.enumerated()), id: \.element.id) { index, asset in
                    Text("图 \(index + 1) · \(asset.name)").tag(Optional(asset.id))
                }
            }.disabled(studio.imageInputsBusy || submissionLocked)
            Picker("输出尺寸", selection: $preset) {
                ForEach(EditingCanvasPreset.allCases) { value in Text(value.title).tag(value) }
            }.pickerStyle(.radioGroup)
                .disabled(studio.imageInputsBusy || submissionLocked)
            if let dimensions = suggestion.dimensions {
                Text("\(dimensions.width) × \(dimensions.height)")
                    .font(.title3.monospacedDigit().weight(.medium))
            }
            Text(suggestion.detail).font(.caption).foregroundStyle(.secondary)
            if let reason = suggestion.blockedReason {
                Label(reason, systemImage: "info.circle").font(.callout).foregroundStyle(.orange)
            }
            if submissionLocked {
                Text("正在准备生成，请等待提交完成。").font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Spacer()
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("应用画布尺寸") {
                    if studio.applyEditingCanvas(preset: preset, assetID: selectedID) { dismiss() }
                }.keyboardShortcut(.defaultAction)
                    .disabled(!suggestion.canApply || submissionLocked)
                    .accessibilityIdentifier("applyEditingCanvas")
            }
        }.padding(24).frame(width: 510)
    }
}
