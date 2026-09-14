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
                print("PASS video preview: mount, decode, playback, stable updates, release and reload; no inference runtime")
                exit(0)
            } catch { print("FAIL \(error)"); exit(1) }
        }
        app.run()
    }
}
