import AppKit
import SwiftUI
import AVKit

/// Run with a real MP4, without loading any model or MLX library.
@main struct VideoPreviewTests {
    @MainActor static func main() {
        guard CommandLine.arguments.count >= 2 else {
            print("video-preview-tests VIDEO.mp4 [UPSCALED.mp4]"); exit(1)
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
                if CommandLine.arguments.count >= 3 {
                    try await compare(source: path, output: CommandLine.arguments[2])
                }
                exit(0)
            } catch { print("FAIL \(error)"); exit(1) }
        }
        app.run()
    }

    @MainActor private static func compare(source: String, output: String) async throws {
        let playback = VideoComparisonPlayback()
        defer { playback.unload() }
        await playback.load(originalPath: source, resultPath: output)
        precondition(playback.size != nil && playback.duration > 0.35, "Comparison could not load the fixture pair")
        precondition(playback.original.isMuted && !playback.result.isMuted, "Both soundtracks could play together")
        playback.toggleAudio()
        precondition(playback.original.isMuted && playback.result.isMuted, "Mute did not affect both sides")

        let initialLeft = playback.original.currentTime().seconds
        let initialRight = playback.result.currentTime().seconds
        func times() -> (left: Double, right: Double) {
            (playback.original.currentTime().seconds - initialLeft,
             playback.result.currentTime().seconds - initialRight)
        }
        func aligned(at expected: Double) -> Bool {
            let (left, right) = times()
            return abs(left - expected) < 0.012 && abs(right - expected) < 0.012
        }
        func waitFor(_ description: String, until condition: () -> Bool) async throws {
            let deadline = ContinuousClock.now + .seconds(4)
            while !condition() {
                precondition(ContinuousClock.now < deadline, description)
                try await Task.sleep(for: .milliseconds(15))
            }
        }

        playback.togglePlayback()
        try await waitFor("The players did not start from the same anchor") {
            let (left, right) = times()
            return min(left, right) > 0.09 && abs(left - right) < 0.012
        }
        playback.togglePlayback()
        let paused = playback.position
        try await waitFor("The players did not pause on the same time") { aligned(at: paused) }
        try await Task.sleep(for: .milliseconds(100))
        precondition(!playback.playing && aligned(at: paused), "One side continued playing while paused")

        playback.togglePlayback()
        playback.setScrubbing(true)
        playback.seek(to: playback.duration * 0.4)
        try await waitFor("The two frames did not follow a drag in progress") { aligned(at: playback.duration * 0.4) }
        playback.seek(to: playback.duration * 0.2)
        playback.seek(to: playback.duration * 0.6)
        let target = playback.duration * 0.6
        try await waitFor("A superseded seek won the scrub queue") { aligned(at: target) }
        precondition(playback.original.rate == 0 && playback.result.rate == 0, "A decoder ran ahead during scrubbing")
        playback.setScrubbing(false)
        try await waitFor("Playback did not resume together after scrubbing") {
            let (left, right) = times()
            return min(left, right) > target + 0.05 && abs(left - right) < 0.012
        }
        try await waitFor("Both sides did not stop at their shared end") {
            !playback.playing && playback.position >= playback.duration && aligned(at: playback.duration)
        }
        playback.togglePlayback()
        try await waitFor("Play did not restart both players from the beginning") {
            let (left, right) = times()
            return min(left, right) > 0.05 && max(left, right) < target && abs(left - right) < 0.012
        }
        playback.unload()
        precondition(playback.original.currentItem == nil && playback.result.currentItem == nil, "Comparison retained a decoder")
        await playback.load(originalPath: source + ".missing", resultPath: output)
        precondition(playback.failure == .original, "A missing history source prevented output-only playback")
        print("PASS video comparison: same-clock play and pause, coalesced accurate scrubbing, synchronized resume and replay, audio and decoder cleanup")
    }
}
