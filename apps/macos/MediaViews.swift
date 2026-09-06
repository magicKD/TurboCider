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
