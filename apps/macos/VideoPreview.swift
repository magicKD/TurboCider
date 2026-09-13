import SwiftUI
import AVKit

/// A concrete AVPlayerView reference also ensures AVKit is directly linked.
/// The old VideoPlayer-only App linked _AVKit_SwiftUI but not AVKit and aborted
/// resolving VideoPlayerView's AVPlayerView superclass on macOS 26.6.2.
/// Keep one item per preview, and release the decoder when it leaves the view.
struct SafeVideoPreview: NSViewRepresentable {
    let path: String
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        Self.setPath(path, on: view)
        return view
    }
    func updateNSView(_ view: AVPlayerView, context: Context) {
        Self.setPath(path, on: view)
    }
    static func setPath(_ path: String, on view: AVPlayerView) {
        let url = URL(fileURLWithPath: path)
        if (view.player?.currentItem?.asset as? AVURLAsset)?.url == url { return }
        releasePlayer(on: view)
        view.player = AVPlayer(url: url)
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
