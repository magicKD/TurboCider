import SwiftUI
import ImageIO

struct Qwen21AnnotationEditor: View {
    let asset: StudioAsset
    @ObservedObject var studio: StudioState
    @Environment(\.dismiss) private var dismiss
    @State private var tool = Qwen21AnnotationTool.ellipse
    @State private var output = Qwen21AnnotationOutput.annotatedImage
    @State private var width = 0.012
    @State private var strokes: [Qwen21AnnotationStroke] = []
    @State private var current: Qwen21AnnotationStroke?
    @State private var preview: NSImage?
    private func path(_ stroke: Qwen21AnnotationStroke, in rect: CGRect) -> Path {
        let points = stroke.points.map { CGPoint(x: rect.minX + $0.x * rect.width, y: rect.minY + $0.y * rect.height) }
        var path = Path()
        guard let a = points.first, let b = points.last else { return path }
        if stroke.tool == .ellipse {
            path.addEllipse(in: CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x-b.x), height: abs(a.y-b.y)))
        } else {
            path.move(to: a)
            if points.count == 1 { path.addLine(to: CGPoint(x: a.x + 0.01, y: a.y)) }
            else { for point in points.dropFirst() { path.addLine(to: point) } }
        }
        return path
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("标注 / 蒙版视觉引导").font(.title2)
            Text("生成新的 PNG（最长边最多 2048），不修改原图。独立蒙版会追加为参考图：白色编辑、黑色保留。均为语义引导，不保证逐像素锁定。")
                .font(.caption).foregroundStyle(.secondary)
            Picker("输出", selection: $output) {
                Text("原图上的红色标注").tag(Qwen21AnnotationOutput.annotatedImage)
                Text("独立黑白蒙版").tag(Qwen21AnnotationOutput.separateMask)
            }.pickerStyle(.segmented).disabled(studio.importing || current != nil)
            HStack {
                Picker("工具", selection: $tool) {
                    Text("圈选").tag(Qwen21AnnotationTool.ellipse)
                    Text("画笔").tag(Qwen21AnnotationTool.brush)
                }.pickerStyle(.segmented).frame(width: 200)
                Slider(value: $width, in: 0.002...0.08).frame(width: 150).accessibilityLabel("标注粗细")
                Button("撤销笔画") { if !strokes.isEmpty { strokes.removeLast() } }.disabled(strokes.isEmpty)
                Button("清空") { strokes.removeAll() }.disabled(strokes.isEmpty)
            }.disabled(studio.importing || current != nil)
            GeometryReader { proxy in
                let ratio = CGFloat(asset.width) / CGFloat(asset.height)
                let w = min(proxy.size.width, proxy.size.height * ratio)
                let h = w / ratio
                let rect = CGRect(x: (proxy.size.width-w)/2, y: (proxy.size.height-h)/2, width: w, height: h)
                ZStack {
                    Color.gray.opacity(0.2)
                    if let preview {
                        Image(nsImage: preview).resizable().frame(width: w, height: h)
                    }
                    Canvas { context, _ in
                        for stroke in strokes + (current.map { [$0] } ?? []) {
                            if output == .separateMask && stroke.tool == .ellipse {
                                context.fill(path(stroke, in: rect), with: .color(.white.opacity(0.65)))
                            } else {
                                context.stroke(path(stroke, in: rect), with: .color(output == .separateMask ? .white.opacity(0.65) : .red),
                                    style: StrokeStyle(lineWidth: stroke.width * min(w, h), lineCap: .round, lineJoin: .round))
                            }
                        }
                    }
                }.contentShape(Rectangle()).gesture(DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        guard !studio.importing, preview != nil, strokes.count < 100 else { return }
                        if current == nil && !rect.contains(drag.startLocation) { return }
                        func normalized(_ point: CGPoint) -> CGPoint {
                            CGPoint(x: max(0, min(1, (point.x-rect.minX)/w)), y: max(0, min(1, (point.y-rect.minY)/h)))
                        }
                        if current == nil { current = Qwen21AnnotationStroke(tool: tool, points: [normalized(drag.startLocation)], width: width) }
                        guard var stroke = current else { return }
                        if stroke.tool == .ellipse { stroke.points = [stroke.points[0], normalized(drag.location)] }
                        else if stroke.points.count < 4096 { stroke.points.append(normalized(drag.location)) }
                        current = stroke
                    }
                    .onEnded { _ in
                        if let stroke = current,
                           stroke.tool != .ellipse || (stroke.points.first!.x != stroke.points.last!.x &&
                                                        stroke.points.first!.y != stroke.points.last!.y) { strokes.append(stroke) }
                        current = nil
                    })
            }.frame(minHeight: 400)
            HStack {
                Text("\(strokes.count) / 100 笔画").font(.caption)
                Spacer()
                Button("取消") { dismiss() }.disabled(studio.importing)
                Button(output == .separateMask ? "添加独立蒙版" : "使用标注副本") {
                    Task { if await studio.annotateQwen21Asset(asset.id, strokes: strokes, output: output) { dismiss() } }
                }.buttonStyle(.borderedProminent).disabled(strokes.isEmpty || current != nil || studio.importing || preview == nil)
            }
            if let message = studio.message { Text(message).font(.caption).foregroundStyle(.secondary) }
        }.padding(20).frame(width: 780, height: 620)
            .onAppear {
                if let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: asset.path) as CFURL, nil),
                   let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: 1600
                   ] as CFDictionary) { preview = NSImage(cgImage: image, size: .zero) }
            }
    }
}
import AppKit
import ImageIO

/// Handle Control-click as selection instead of macOS's usual secondary click.
/// A physical right-click can still open the surrounding context menu.
struct ResultSelectionTarget: NSViewRepresentable {
    let label: String
    let selected: Bool
    let action: (NSEvent.ModifierFlags, Int) -> Void

    func makeNSView(context: Context) -> ResultSelectionView { ResultSelectionView() }
    func updateNSView(_ view: ResultSelectionView, context: Context) {
        view.action = action
        view.setAccessibilityElement(true)
        view.setAccessibilityRole(.button)
        view.setAccessibilityLabel(label)
        view.setAccessibilityValue(selected ? "已选择" : "未选择")
    }
}

final class ResultSelectionView: NSView {
    var action: ((NSEvent.ModifierFlags, Int) -> Void)?
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func mouseDown(with event: NSEvent) { action?(event.modifierFlags, event.clickCount) }
    override func accessibilityPerformPress() -> Bool { action?([], 1); return true }
}

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
