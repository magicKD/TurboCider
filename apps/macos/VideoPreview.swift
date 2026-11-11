import SwiftUI
import AVKit

/// Only decoded stills are cached. History grids never retain video players or
/// full-resolution frames, and replacing a file at the same path invalidates it.
actor VideoThumbnailCache {
    static let shared = VideoThumbnailCache()
    private let images = NSCache<NSString, CGImage>()

    init() {
        images.countLimit = 96
        images.totalCostLimit = 32 * 1024 * 1024
    }

    func thumbnail(path: String, maxPixel: Int = 320) async throws -> CGImage {
        try Task.checkCancellation()
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let values = try url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let limit = min(2400, max(1, maxPixel))
        let key = "\(url.path)#\(values.contentModificationDate?.timeIntervalSince1970 ?? 0)#\(values.fileSize ?? 0)#\(limit)" as NSString
        if let image = images.object(forKey: key) { return image }
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: limit, height: limit)
        let image = try await withTaskCancellationHandler {
            try await generator.image(at: .zero).image
        } onCancel: {
            generator.cancelAllCGImageGeneration()
        }
        try Task.checkCancellation()
        images.setObject(image, forKey: key, cost: image.bytesPerRow * image.height)
        return image
    }
}

struct VideoThumbnail: View {
    let path: String
    var maxPixel = 320
    var showsBadge = true
    @State private var image: NSImage?
    @State private var failed = false

    var body: some View {
        Group {
            if let image { Image(nsImage: image).resizable().scaledToFit() }
            else if failed {
                Label("无法读取视频", systemImage: "film").font(.caption).foregroundStyle(.secondary)
            } else { ProgressView().controlSize(.small) }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.primary.opacity(0.06))
            .overlay(alignment: .bottomTrailing) {
                if showsBadge {
                    Image(systemName: "play.fill").font(.system(size: 10)).foregroundStyle(.white)
                        .padding(5).background(.black.opacity(0.55), in: Circle()).padding(4)
                }
            }
            .task(id: "\(path)#\(maxPixel)") {
                image = nil; failed = false
                do {
                    let decoded = try await VideoThumbnailCache.shared.thumbnail(path: path, maxPixel: maxPixel)
                    guard !Task.isCancelled else { return }
                    image = NSImage(cgImage: decoded, size: .zero)
                } catch {
                    guard !Task.isCancelled else { return }
                    failed = true
                }
            }
    }
}

/// Show the actual first frame immediately; create a player only on play.
struct VideoOutputPreview: View {
    let path: String
    @State private var playing = false

    var body: some View {
        Group {
            if playing { SafeVideoPreview(path: path, autoplay: true) }
            else {
                Button { playing = true } label: {
                    VideoThumbnail(path: path, maxPixel: 1600, showsBadge: false)
                        .overlay {
                            Image(systemName: "play.circle.fill").font(.system(size: 54))
                                .foregroundStyle(.white).shadow(radius: 6)
                        }.contentShape(Rectangle())
                }.buttonStyle(.plain).accessibilityLabel("播放视频").accessibilityIdentifier("playVideo")
            }
        }.onChange(of: path) { _, _ in playing = false }
    }
}

/// A concrete AVPlayerView reference also ensures AVKit is directly linked.
/// The old VideoPlayer-only App linked _AVKit_SwiftUI but not AVKit and aborted
/// resolving VideoPlayerView's AVPlayerView superclass on macOS 26.6.2.
/// Keep one item per preview, and release the decoder when it leaves the view.
struct SafeVideoPreview: NSViewRepresentable {
    let path: String
    var autoplay = false
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        Self.setPath(path, on: view, autoplay: autoplay)
        return view
    }
    func updateNSView(_ view: AVPlayerView, context: Context) {
        Self.setPath(path, on: view, autoplay: autoplay)
    }
    static func setPath(_ path: String, on view: AVPlayerView, autoplay: Bool = false) {
        let url = URL(fileURLWithPath: path)
        if (view.player?.currentItem?.asset as? AVURLAsset)?.url == url { return }
        releasePlayer(on: view)
        view.player = AVPlayer(url: url)
        if autoplay { view.player?.play() }
    }
    static func releasePlayer(on view: AVPlayerView) {
        view.player?.pause()
        view.player?.replaceCurrentItem(with: nil)
        view.player = nil
    }
    static func dismantleNSView(_ view: AVPlayerView, coordinator: ()) {
        releasePlayer(on: view)
    }
}
