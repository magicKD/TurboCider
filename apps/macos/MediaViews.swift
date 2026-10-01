import SwiftUI
import AppKit
import ImageIO

/// One model-encoding selector shared by Creation and independent Playground
/// drafts. Choosing standard is always available as explicit recovery.
struct Qwen21ReferenceEncodingSettings: View {
    let draft: StudioDraft
    var locked = false
    var accessibilityPrefix = "qwen21"
    let setSize: (Int) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("模型参考编码").font(.caption.weight(.semibold))
            Picker("编码尺度", selection: Binding(get: { draft.qwen21ReferenceSize }, set: setSize)) {
                Text("标准 1024").tag(1024)
                Text("快速 512（预览）").tag(512).disabled(draft.qwen21FastReferenceUnavailableReason != nil)
                if ![1024, 512].contains(draft.qwen21ReferenceSize) {
                    Text("未支持 \(draft.qwen21ReferenceSize)").tag(draft.qwen21ReferenceSize).disabled(true)
                }
            }.labelsHidden().disabled(locked).accessibilityIdentifier(accessibilityPrefix + "ReferenceEncoding")
            Text("按模型比例编码，与输入文件尺寸独立。快速 512 是近似预览，可能丢失参考细节；精细脸部、文字建议标准 1024。")
                .font(.caption2).foregroundStyle(.secondary)
            if let issue = draft.qwen21ReferenceSizeIssue {
                Text(issue).font(.caption2).foregroundStyle(.orange).textSelection(.enabled)
                    .accessibilityIdentifier(accessibilityPrefix + "ReferenceEncodingIssue")
                Button("恢复标准 1024") { setSize(1024) }.font(.caption).disabled(locked)
                    .accessibilityIdentifier(accessibilityPrefix + "ReferenceEncodingReset")
            } else if let reason = draft.qwen21FastReferenceUnavailableReason {
                Text(reason).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

private struct ImageViewport {
    var zoom: CGFloat = 1
    var offset = CGSize.zero

    func imageRect(source: CGSize, bounds: CGSize) -> CGRect {
        let fit = min(max(0, bounds.width) / max(1, source.width), max(0, bounds.height) / max(1, source.height))
        let size = CGSize(width: source.width * fit * zoom, height: source.height * fit * zoom)
        let position = clamped(offset, image: size, bounds: bounds)
        return CGRect(x: (bounds.width - size.width) / 2 + position.width,
                      y: (bounds.height - size.height) / 2 + position.height,
                      width: size.width, height: size.height)
    }
    mutating func setZoom(_ value: CGFloat, source: CGSize, bounds: CGSize) {
        guard value.isFinite else { return }
        let next = min(8, max(1, value))
        offset = CGSize(width: offset.width * next / zoom, height: offset.height * next / zoom)
        zoom = next
        constrain(source: source, bounds: bounds)
    }
    mutating func pan(from origin: CGSize, by translation: CGSize, source: CGSize, bounds: CGSize) {
        offset = CGSize(width: origin.width + translation.width, height: origin.height + translation.height)
        constrain(source: source, bounds: bounds)
    }
    mutating func constrain(source: CGSize, bounds: CGSize) {
        offset = clamped(offset, image: imageRect(source: source, bounds: bounds).size, bounds: bounds)
    }
    private func clamped(_ position: CGSize, image: CGSize, bounds: CGSize) -> CGSize {
        let horizontal = max(0, (image.width - bounds.width) / 2)
        let vertical = max(0, (image.height - bounds.height) / 2)
        return CGSize(width: min(horizontal, max(-horizontal, position.width)),
                      height: min(vertical, max(-vertical, position.height)))
    }
}

private struct ImageZoomControls: View {
    let zoom: CGFloat
    let fit: () -> Void
    let change: (CGFloat) -> Void
    var body: some View {
        HStack(spacing: 8) {
            Button(action: fit) { Image(systemName: "arrow.down.right.and.arrow.up.left") }
                .help("适应窗口")
                .accessibilityLabel("适应窗口").accessibilityIdentifier("imageFit")
            Divider().frame(height: 16)
            Button { change(zoom / 1.5) } label: { Image(systemName: "minus.magnifyingglass") }
                .disabled(zoom <= 1).help("缩小").accessibilityLabel("缩小图片").accessibilityIdentifier("imageZoomOut")
            Text(zoom <= 1 ? "适应" : "\(Int((zoom * 100).rounded()))%")
                .font(.caption.monospacedDigit()).frame(minWidth: 42)
                .help("相对于适应窗口的倍率").accessibilityIdentifier("imageZoom")
            Button { change(zoom * 1.5) } label: { Image(systemName: "plus.magnifyingglass") }
                .disabled(zoom >= 8).help("放大；放大后拖动查看细节")
                .accessibilityLabel("放大图片").accessibilityIdentifier("imageZoomIn")
        }.buttonStyle(.borderless).padding(.horizontal, 12).padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
            .overlay(Capsule().stroke(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

/// A fitted image canvas with local zoom/pan state. Changing the path resets the
/// viewport; decoding remains off the main actor and cancelled loads cannot publish.
struct InteractiveMediaPreview: View {
    let path: String
    var maxPixel = 2400
    var showsControls = true
    @State private var image: NSImage?
    @State private var failed = false
    @State private var transparent = false
    @State private var viewport = ImageViewport()
    @State private var panOrigin: CGSize?
    @State private var magnificationOrigin: ImageViewport?

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.primary.opacity(0.025)
                if let image {
                    let rect = viewport.imageRect(source: image.size, bounds: proxy.size)
                    Image(nsImage: image).resizable().interpolation(.high)
                        .frame(width: rect.width, height: rect.height)
                        .background { if transparent { TransparencyGrid() } }
                        .position(x: rect.midX, y: rect.midY)
                } else if failed {
                    Label("无法读取图片", systemImage: "photo.badge.exclamationmark")
                        .font(.callout).foregroundStyle(.secondary)
                } else { ProgressView().controlSize(.small) }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .clipped().contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 2)
                .onChanged { drag in
                    guard let image, viewport.zoom > 1, magnificationOrigin == nil else { return }
                    if panOrigin == nil { panOrigin = viewport.offset }
                    viewport.pan(from: panOrigin ?? .zero, by: drag.translation, source: image.size, bounds: proxy.size)
                }
                .onEnded { _ in panOrigin = nil })
            .simultaneousGesture(MagnificationGesture()
                .onChanged { value in
                    guard let image else { return }
                    if magnificationOrigin == nil { magnificationOrigin = viewport; panOrigin = nil }
                    var next = magnificationOrigin ?? viewport
                    next.setZoom(next.zoom * value, source: image.size, bounds: proxy.size)
                    viewport = next
                }
                .onEnded { _ in magnificationOrigin = nil; panOrigin = nil })
            .onTapGesture(count: 2) { viewport = ImageViewport(); panOrigin = nil; magnificationOrigin = nil }
            .overlay(alignment: .bottomTrailing) {
                if showsControls, let image {
                    ImageZoomControls(zoom: viewport.zoom,
                                      fit: { viewport = ImageViewport(); panOrigin = nil; magnificationOrigin = nil },
                                      change: { viewport.setZoom($0, source: image.size, bounds: proxy.size) })
                        .padding(12)
                }
            }
            .onChange(of: proxy.size) { _, size in
                if let image { viewport.constrain(source: image.size, bounds: size) }
                panOrigin = nil; magnificationOrigin = nil
            }
        }.accessibilityIdentifier("interactiveMediaPreview")
            .task(id: "\(path)#\(maxPixel)") {
                image = nil; failed = false; transparent = false
                viewport = ImageViewport(); panOrigin = nil; magnificationOrigin = nil
                let filePath = path, pixelLimit = max(1, maxPixel)
                let decoded = await Task.detached(priority: .utility) { () -> CGImage? in
                    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: filePath) as CFURL, nil) else { return nil }
                    return CGImageSourceCreateThumbnailAtIndex(source, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: pixelLimit
                    ] as CFDictionary)
                }.value
                guard !Task.isCancelled else { return }
                image = decoded.map { NSImage(cgImage: $0, size: .zero) }; failed = decoded == nil
                if let decoded { transparent = ![CGImageAlphaInfo.none, .noneSkipFirst, .noneSkipLast].contains(decoded.alphaInfo) }
            }
    }
}

private enum AnnotationInteractionTool: String { case ellipse, brush, pan }

struct Qwen21AnnotationEditor: View {
    let asset: StudioAsset
    @ObservedObject var studio: StudioState
    @Environment(\.dismiss) private var dismiss
    @State private var tool = AnnotationInteractionTool.ellipse
    @State private var output = Qwen21AnnotationOutput.annotatedImage
    @State private var width = 0.012
    @State private var strokes: [Qwen21AnnotationStroke] = []
    @State private var current: Qwen21AnnotationStroke?
    @State private var preview: NSImage?
    @State private var previewError: String?
    @State private var viewport = ImageViewport()
    @State private var panOrigin: CGSize?
    @State private var magnificationOrigin: ImageViewport?
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
            HStack {
                Text("编辑参考图").font(.title2.weight(.medium))
                Spacer()
                Text("\(asset.width) × \(asset.height)").font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
            Text(asset.name).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            Picker("输出", selection: $output) {
                Text("红色标注副本").tag(Qwen21AnnotationOutput.annotatedImage)
                Text("独立黑白蒙版").tag(Qwen21AnnotationOutput.separateMask)
            }.pickerStyle(.segmented).disabled(studio.importing || current != nil)
            HStack {
                Picker("工具", selection: $tool) {
                    Text("圈选").tag(AnnotationInteractionTool.ellipse)
                    Text("画笔").tag(AnnotationInteractionTool.brush)
                    Text("移动").tag(AnnotationInteractionTool.pan)
                }.pickerStyle(.segmented).frame(width: 240).accessibilityIdentifier("annotationTool")
                Text("粗细").font(.caption).foregroundStyle(.secondary)
                Slider(value: $width, in: 0.002...0.08).frame(width: 130).disabled(tool == .pan).accessibilityLabel("标注粗细")
                Spacer()
                Button { if !strokes.isEmpty { strokes.removeLast() } } label: { Image(systemName: "arrow.uturn.backward") }
                    .disabled(strokes.isEmpty).help("撤销笔画").accessibilityLabel("撤销笔画")
                Button("清空") { strokes.removeAll() }.disabled(strokes.isEmpty)
            }.disabled(studio.importing || current != nil)
            GeometryReader { proxy in
                let sourceSize = CGSize(width: CGFloat(max(1, asset.width)), height: CGFloat(max(1, asset.height)))
                let rect = viewport.imageRect(source: sourceSize, bounds: proxy.size)
                ZStack {
                    Color.primary.opacity(0.025)
                    if let preview {
                        Image(nsImage: preview).resizable().frame(width: rect.width, height: rect.height)
                            .opacity(output == .separateMask ? 0.3 : 1)
                            .background {
                                if output == .separateMask { Color.black }
                                else { TransparencyGrid() }
                            }
                            .position(x: rect.midX, y: rect.midY)
                    } else if previewError == nil {
                        ProgressView("正在读取原图…")
                    }
                    Canvas { context, _ in
                        context.clip(to: Path(rect))
                        for stroke in strokes + (current.map { [$0] } ?? []) {
                            if output == .separateMask && stroke.tool == .ellipse {
                                context.fill(path(stroke, in: rect), with: .color(.white))
                            } else {
                                context.stroke(path(stroke, in: rect), with: .color(output == .separateMask ? .white : .red),
                                    style: StrokeStyle(lineWidth: stroke.width * min(rect.width, rect.height), lineCap: .round, lineJoin: .round))
                            }
                        }
                    }
                }.frame(width: proxy.size.width, height: proxy.size.height).clipped()
                    .contentShape(Rectangle()).gesture(DragGesture(minimumDistance: 0)
                    .onChanged { drag in
                        guard !studio.importing, preview != nil, magnificationOrigin == nil else { return }
                        if tool == .pan {
                            if panOrigin == nil { panOrigin = viewport.offset }
                            viewport.pan(from: panOrigin ?? .zero, by: drag.translation, source: sourceSize, bounds: proxy.size)
                            return
                        }
                        guard strokes.count < 100, rect.width > 0, rect.height > 0 else { return }
                        if current == nil && !rect.contains(drag.startLocation) { return }
                        func normalized(_ point: CGPoint) -> CGPoint {
                            CGPoint(x: max(0, min(1, (point.x-rect.minX)/rect.width)), y: max(0, min(1, (point.y-rect.minY)/rect.height)))
                        }
                        if current == nil { current = Qwen21AnnotationStroke(tool: tool == .brush ? .brush : .ellipse, points: [normalized(drag.startLocation)], width: width) }
                        guard var stroke = current else { return }
                        if stroke.tool == .ellipse { stroke.points = [stroke.points[0], normalized(drag.location)] }
                        else if stroke.points.count < 4096 { stroke.points.append(normalized(drag.location)) }
                        current = stroke
                    }
                    .onEnded { _ in
                        panOrigin = nil
                        if let stroke = current, let first = stroke.points.first, let last = stroke.points.last,
                           stroke.tool != .ellipse || (first.x != last.x && first.y != last.y) { strokes.append(stroke) }
                        current = nil
                    })
                    .simultaneousGesture(MagnificationGesture()
                        .onChanged { value in
                            guard !studio.importing, current == nil else { return }
                            if magnificationOrigin == nil { magnificationOrigin = viewport; panOrigin = nil }
                            var next = magnificationOrigin ?? viewport
                            next.setZoom(next.zoom * value, source: sourceSize, bounds: proxy.size)
                            viewport = next
                        }
                        .onEnded { _ in magnificationOrigin = nil; panOrigin = nil })
                    .overlay(alignment: .bottomTrailing) {
                        ImageZoomControls(zoom: viewport.zoom, fit: { viewport = ImageViewport(); panOrigin = nil; magnificationOrigin = nil },
                                          change: { viewport.setZoom($0, source: sourceSize, bounds: proxy.size) })
                            .disabled(studio.importing || current != nil || preview == nil).padding(12)
                    }
                    .onChange(of: proxy.size) { _, size in viewport.constrain(source: sourceSize, bounds: size); panOrigin = nil }
            }.frame(minHeight: 280, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 10))
            Text(tool == .pan ? "放大后拖动画面；适应窗口恢复全图。" : output == .separateMask
                 ? "白色为编辑区；淡化原图仅用于定位。可先放大，再圈选或涂抹。" : "在图中拖动圈选或涂抹，生成红色标注副本。")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Text("\(strokes.count) / 100 笔画").font(.caption).foregroundStyle(.secondary)
                if studio.importing { ProgressView().controlSize(.small) }
                Spacer()
                Button("取消") { dismiss() }.disabled(studio.importing).keyboardShortcut(.cancelAction)
                Button(output == .separateMask ? "添加独立蒙版" : "使用标注副本") {
                    Task { if await studio.annotateQwen21Asset(asset.id, strokes: strokes, output: output) { dismiss() } }
                }.buttonStyle(.borderedProminent).disabled(strokes.isEmpty || current != nil || studio.importing || preview == nil)
                    .keyboardShortcut(.return, modifiers: .command)
            }
            Text("新 PNG 最长边 2048，不修改原图。蒙版作为额外参考图提供语义引导，不保证逐像素锁定。")
                .font(.caption2).foregroundStyle(.secondary)
            if let previewError { Label(previewError, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.red) }
            if let message = studio.message { ScrollView { Text(message).font(.caption).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(maxHeight: 44) }
        }.padding(20).frame(minWidth: 740, idealWidth: 980, maxWidth: .infinity, minHeight: 580, idealHeight: 740, maxHeight: .infinity)
            .task(id: asset.path) {
                viewport = ImageViewport(); current = nil; panOrigin = nil; magnificationOrigin = nil
                preview = nil; previewError = nil; strokes = []
                let path = asset.path
                let image = await Task.detached(priority: .utility) { () -> CGImage? in
                    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
                    return CGImageSourceCreateThumbnailAtIndex(source, 0, [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                        kCGImageSourceThumbnailMaxPixelSize: 2400
                    ] as CFDictionary)
                }.value
                guard !Task.isCancelled else { return }
                preview = image.map { NSImage(cgImage: $0, size: .zero) }
                previewError = image == nil ? "无法读取原图，请关闭标注窗口并重新导入图片。" : nil
            }
    }
}

private struct TransparencyGrid: View {
    @Environment(\.colorScheme) private var colorScheme
    private static func pattern(background: NSColor, square: NSColor) -> NSImage {
        NSImage(size: NSSize(width: 24, height: 24), flipped: false) { bounds in
            background.setFill(); NSBezierPath(rect: bounds).fill()
            square.setFill()
            NSBezierPath(rect: NSRect(x: 0, y: 0, width: 12, height: 12)).fill()
            NSBezierPath(rect: NSRect(x: 12, y: 12, width: 12, height: 12)).fill()
            return true
        }
    }
    private static let lightPattern = pattern(background: NSColor(calibratedWhite: 0.96, alpha: 1),
                                              square: NSColor(calibratedWhite: 0.89, alpha: 1))
    private static let darkPattern = pattern(background: NSColor(calibratedWhite: 0.18, alpha: 1),
                                             square: NSColor(calibratedWhite: 0.25, alpha: 1))
    var body: some View {
        Canvas { context, size in
            // One tiled fill keeps a deeply zoomed transparent image inexpensive.
            let tile = colorScheme == .dark ? Self.darkPattern : Self.lightPattern
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .tiledImage(Image(nsImage: tile)))
        }.accessibilityHidden(true).allowsHitTesting(false)
    }
}

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
    @State private var transparent = false
    var body: some View {
        Group {
            if let image {
                Image(nsImage: image).resizable().scaledToFit()
                    .background { if transparent { TransparencyGrid() } }
            }
            else if failed { Label("无法读取图片", systemImage: "photo.badge.exclamationmark").font(.caption).foregroundStyle(.secondary) }
            else { ProgressView().controlSize(.small) }
        }.task(id: "\(path)#\(maxPixel)") {
            image = nil; failed = false; transparent = false
            let pixelLimit = maxPixel, filePath = path
            let decoded = await Task.detached(priority: .utility) { () -> CGImage? in
                guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: filePath) as CFURL, nil) else { return nil }
                return CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: pixelLimit] as CFDictionary)
            }.value
            guard !Task.isCancelled else { return }
            image = decoded.map { NSImage(cgImage: $0, size: .zero) }; failed = decoded == nil
            if let decoded {
                transparent = ![CGImageAlphaInfo.none, .noneSkipFirst, .noneSkipLast].contains(decoded.alphaInfo)
            }
        }
    }
}

/// Plain-text paste stays in the prompt. Image-only paste is handed to the input
/// workspace, without changing the insertion point or pasting rich HTML.
private final class PromptTextView: NSTextView {
    var pasteImage: (() -> Void)?
    var placeholder = "" { didSet { needsDisplay = true } }
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard string.isEmpty, !placeholder.isEmpty else { return }
        (placeholder as NSString).draw(at: NSPoint(x: textContainerInset.width + 5, y: textContainerInset.height),
                                      withAttributes: [.font: font ?? NSFont.systemFont(ofSize: 14),
                                                       .foregroundColor: NSColor.placeholderTextColor])
    }
    override func didChangeText() { super.didChangeText(); needsDisplay = true }
    override func paste(_ sender: Any?) {
        guard isEditable else { return }
        let board = NSPasteboard.general
        if board.string(forType: .string) == nil && (board.canReadObject(forClasses: [NSImage.self], options: nil) || board.availableType(from: [.fileURL]) != nil) {
            pasteImage?()
        } else { pasteAsPlainText(sender) }
    }
}
struct PromptEditor: NSViewRepresentable {
    @Binding var text: String
    var placeholder = ""
    var editable = true
    let pasteImage: () -> Void
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView(); scroll.hasVerticalScroller = true; scroll.drawsBackground = false
        let editor = PromptTextView()
        editor.isRichText = false; editor.isAutomaticQuoteSubstitutionEnabled = false
        editor.isAutomaticDashSubstitutionEnabled = false
        editor.isAutomaticTextReplacementEnabled = false
        editor.font = .systemFont(ofSize: 14); editor.textColor = .labelColor; editor.drawsBackground = false
        editor.isVerticallyResizable = true; editor.isHorizontallyResizable = false
        editor.autoresizingMask = [.width]; editor.textContainer?.widthTracksTextView = true
        editor.textContainerInset = NSSize(width: 2, height: 4)
        editor.delegate = context.coordinator; editor.string = text; editor.pasteImage = pasteImage
        editor.placeholder = placeholder; editor.isEditable = editable
        editor.setAccessibilityIdentifier("prompt"); editor.setAccessibilityLabel("提示词")
        scroll.documentView = editor
        return scroll
    }
    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let editor = scroll.documentView as? PromptTextView else { return }
        if editor.string != text && !editor.hasMarkedText() { editor.string = text; editor.needsDisplay = true }
        editor.pasteImage = pasteImage; editor.placeholder = placeholder; editor.isEditable = editable
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
