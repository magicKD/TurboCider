import AppKit
import SwiftUI
import AVKit

/// Run with a real MP4, without loading any model or MLX library.
@main struct VideoPreviewTests {
    @MainActor static func main() {
        guard CommandLine.arguments.count >= 2 else {
            print("video-preview-tests VIDEO.mp4"); exit(1)
        }
        let path = CommandLine.arguments[1]
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 704, height: 448),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: SafeVideoPreview(path: path))
        window.orderFrontRegardless()
        Task { @MainActor in
            do {
                let thumbnails = VideoThumbnailCache()
                let first = try await thumbnails.thumbnail(path: path, maxPixel: 120)
                precondition(first.width > 0 && first.height > 0 && max(first.width, first.height) <= 120,
                             "Video thumbnail ignored the requested pixel bound")
                let cached = try await thumbnails.thumbnail(path: path, maxPixel: 120)
                precondition(first === cached, "Repeated history previews did not reuse their thumbnail")
                let larger = try await thumbnails.thumbnail(path: path, maxPixel: 320)
                precondition(max(larger.width, larger.height) <= 320, "Large thumbnail exceeded its bound")
                let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-video-preview-\(UUID())")
                try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                defer { try? FileManager.default.removeItem(at: root) }
                let replaced = root.appendingPathComponent("replaced.mp4")
                try FileManager.default.copyItem(at: URL(fileURLWithPath: path), to: replaced)
                _ = try await thumbnails.thumbnail(path: replaced.path, maxPixel: 120)
                try Data("not a video".utf8).write(to: replaced, options: .atomic)
                do {
                    _ = try await thumbnails.thumbnail(path: replaced.path, maxPixel: 120)
                    fatalError("Replacing the file reused a stale successful thumbnail")
                } catch { /* An unreadable replacement must fail, not show stale media. */ }
                try FileManager.default.removeItem(at: replaced)
                try FileManager.default.copyItem(at: URL(fileURLWithPath: path), to: replaced)
                _ = try await thumbnails.thumbnail(path: replaced.path, maxPixel: 120)
                let cancelled = Task {
                    withUnsafeCurrentTask { $0?.cancel() }
                    return try await thumbnails.thumbnail(path: path, maxPixel: 120)
                }
                do {
                    _ = try await cancelled.value
                    fatalError("A cancelled thumbnail request returned cached media")
                } catch is CancellationError { }
                try await Task.sleep(for: .seconds(2))
                let asset = AVURLAsset(url: URL(fileURLWithPath: path))
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard !tracks.isEmpty else { fatalError("No video track") }
                let generator = AVAssetImageGenerator(asset: asset)
                _ = try await generator.image(at: .zero)
                do {
                    @MainActor func playerView(_ view: NSView) -> AVPlayerView? {
                        if let player = view as? AVPlayerView { return player }
                        return view.subviews.lazy.compactMap { playerView($0) }.first
                    }
                    guard let view = window.contentView.flatMap(playerView), let player = view.player else {
                        fatalError("SwiftUI did not mount the AppKit video player")
                    }
                    SafeVideoPreview.setPath(path, on: view)
                    precondition(view.player === player, "A refresh recreated the player")
                    player.play()
                    try await Task.sleep(for: .seconds(1))
                    precondition(player.currentTime().seconds > 0, "Playback did not advance")
                    SafeVideoPreview.releasePlayer(on: view)
                    precondition(view.player == nil && player.currentItem == nil, "Preview retained its media")
                    SafeVideoPreview.setPath(path, on: view)
                    precondition(view.player !== player, "Preview failed to reload")
                }
                window.contentView = NSView()
                window.close()
                print("PASS video preview: bounded/cached thumbnails, file replacement, corrupt-file recovery, cancellation, mount, playback, stable updates, release and reload; no inference runtime")
                exit(0)
            } catch { print("FAIL \(error)"); exit(1) }
        }
        app.run()
    }
}
