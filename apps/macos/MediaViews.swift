import SwiftUI
import AppKit
import ImageIO

/// Decoding happens once per path/size, never on telemetry updates.
struct MediaPreview: View {
    let path: String
    var maxPixel = 1600
    @State private var image: NSImage?
    @State private var failed = false
    var body: some View {
        Group {
            if let image { Image(nsImage: image).resizable().scaledToFit() }
            else if failed { Label("无法读取图片", systemImage: "photo.badge.exclamationmark").font(.caption).foregroundStyle(.secondary) }
            else { ProgressView().controlSize(.small) }
        }.task(id: "\(path)#\(maxPixel)") {
            image = nil; failed = false
            let pixelLimit = maxPixel, filePath = path
            let decoded = await Task.detached(priority: .utility) { () -> CGImage? in
                guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: filePath) as CFURL, nil) else { return nil }
                return CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: pixelLimit] as CFDictionary)
            }.value
            guard !Task.isCancelled else { return }
            image = decoded.map { NSImage(cgImage: $0, size: .zero) }; failed = decoded == nil
        }
    }
}

/// Plain-text paste stays in the prompt. Image-only paste is handed to the input
/// workspace, without changing the insertion point or pasting rich HTML.
private final class PromptTextView: NSTextView {
    var pasteImage: (() -> Void)?
    override func paste(_ sender: Any?) {
        let board = NSPasteboard.general
        if board.string(forType: .string) == nil && (board.canReadObject(forClasses: [NSImage.self], options: nil) || board.availableType(from: [.fileURL]) != nil) {
            pasteImage?()
        } else { pasteAsPlainText(sender) }
    }
}
struct PromptEditor: NSViewRepresentable {
    @Binding var text: String
    let pasteImage: () -> Void
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView(); scroll.hasVerticalScroller = true; scroll.drawsBackground = false
        let editor = PromptTextView()
        editor.isRichText = false; editor.isAutomaticQuoteSubstitutionEnabled = false
        editor.font = .systemFont(ofSize: 14); editor.textColor = .labelColor; editor.drawsBackground = false
        editor.isVerticallyResizable = true; editor.isHorizontallyResizable = false
        editor.autoresizingMask = [.width]; editor.textContainer?.widthTracksTextView = true
        editor.textContainerInset = NSSize(width: 2, height: 4)
        editor.delegate = context.coordinator; editor.string = text; editor.pasteImage = pasteImage
        editor.setAccessibilityIdentifier("prompt"); editor.setAccessibilityLabel("提示词")
        scroll.documentView = editor
        return scroll
    }
    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let editor = scroll.documentView as? PromptTextView else { return }
        if editor.string != text && !editor.hasMarkedText() { editor.string = text }
        editor.pasteImage = pasteImage
    }
    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptEditor
        init(_ parent: PromptEditor) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            if let view = notification.object as? NSTextView { parent.text = view.string }
        }
    }
}

/// Count with the native generation tokenizer; characters are never presented as tokens.
struct PromptCapacityView: View {
    @ObservedObject var studio: StudioState
    var busy: Bool
    @State private var countedKey: [String] = []
    @State private var tokens: Int?
    @State private var countError: String?
    private var key: [String] { [studio.draft.modelID, studio.draft.modelPath, studio.draft.prompt] }
    private var count: Int? { countedKey == key ? tokens : nil }
    private var isZImage: Bool { studio.draft.modelID == "z-image-turbo" }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("\(studio.draft.prompt.count) 字符")
                if isZImage, let count {
                    Text("· \(count) / 1024 tokens（含模板）")
                        .foregroundStyle(count > 1024 ? Color.red : Color.secondary)
                } else if isZImage, !studio.draft.prompt.isEmpty {
                    Text(countedKey == key && countError != nil ? "· token 计数不可用" : "· 正在计数…")
                }
                Spacer()
                if studio.draft.usesANE {
                    Button("改用 GPU") { studio.setANEEnabled(false) }.disabled(busy)
                        .accessibilityIdentifier("promptUseGPU")
                }
            }.accessibilityIdentifier("promptLength")
            if isZImage, let count {
                if count > 1024 {
                    Text("超过当前上限，请缩短文本。GPU 与 ANE 上限相同，不会自动截断。")
                        .foregroundStyle(.red)
                } else if count > 512 {
                    Text("长文本扩展：超过常用的 512 tokens，耗时与内存会增加，描述遵循效果需实际验证。")
                }
                if studio.draft.usesANE, count <= 1024 {
                    let textRows = (count + 31) / 32 * 32
                    let imageRows = ((studio.draft.width / 16) * (studio.draft.height / 16) + 31) / 32 * 32
                    Text("ANE 需要 \(imageRows + textRows) 行分区；容量不足或运行较慢时可改用 GPU。")
                }
            }
            if countedKey == key, let countError { Text(countError).foregroundStyle(.orange) }
        }.font(.caption2).foregroundStyle(.secondary)
        .task(id: key) {
            let requested = key
            guard isZImage, !requested[2].isEmpty else {
                tokens = nil; countError = nil; countedKey = requested; return
            }
            do {
                try await Task.sleep(for: .milliseconds(350))
                let result = try await Task.detached(priority: .utility) {
                    try NativeEngine.zImageTokenCount(modelPath: requested[1], prompt: requested[2])
                }.value
                try Task.checkCancellation()
                tokens = result; countError = nil; countedKey = requested
            } catch is CancellationError {
                // A newer edit owns the counter now.
            } catch {
                guard !Task.isCancelled else { return }
                tokens = nil; countError = error.localizedDescription; countedKey = requested
            }
        }
    }
}
