import SwiftUI
import AVKit

/// Both decoders play from the same host-time anchor. A single queue coalesces
/// scrub requests and waits for both precise seeks before resuming playback.
@MainActor final class VideoComparisonPlayback: ObservableObject {
    enum Failure: Error {
        case original, result, timeline
    }

    private struct Clip {
        let asset: AVURLAsset
        let start: CMTime
        let length: CMTime
        let size: CGSize
        let hasAudio: Bool

        static func load(_ path: String, failure: Failure) async throws -> Clip {
            let asset = AVURLAsset(url: URL(fileURLWithPath: path),
                                   options: [AVURLAssetPreferPreciseDurationAndTimingKey: true])
            do {
                guard try await asset.load(.isPlayable),
                      let track = try await asset.loadTracks(withMediaType: .video).first else { throw failure }
                let (range, segments, transform, naturalSize, formats) = try await track.load(
                    .timeRange, .segments, .preferredTransform, .naturalSize, .formatDescriptions)
                // An initial empty edit carries no picture. Align the first
                // visible frames even when only one export retains that edit.
                let firstPicture = segments.first { !$0.isEmpty }?.timeMapping.target.start ?? range.start
                let start = firstPicture.isNumeric ? CMTimeMaximum(range.start, firstPicture) : range.start
                let length = CMTimeSubtract(range.end, start)
                let displaySize = formats.first.map {
                    CMVideoFormatDescriptionGetPresentationDimensions($0, usePixelAspectRatio: true, useCleanAperture: true)
                } ?? naturalSize
                let size = CGRect(origin: .zero, size: displaySize).applying(transform).size
                guard start.isNumeric, length.isNumeric, length.seconds > 0,
                      size.width.isFinite, size.height.isFinite, size.width > 0, size.height > 0 else { throw failure }
                let audio = failure == .result ? (try? await asset.loadTracks(withMediaType: .audio)) ?? [] : []
                return Clip(asset: asset, start: start, length: length, size: size, hasAudio: !audio.isEmpty)
            } catch { throw failure }
        }
    }

    let original = AVPlayer()
    let result = AVPlayer()
    @Published private(set) var size: CGSize?
    @Published private(set) var duration = 0.0
    @Published private(set) var position = 0.0
    @Published private(set) var playing = false
    @Published private(set) var failure: Failure?
    @Published private(set) var playbackInterrupted = false
    @Published private(set) var hasAudio = false
    @Published private(set) var muted = false

    private let hostClock = CMClockGetHostTimeClock()
    private var originalStart = CMTime.zero
    private var resultStart = CMTime.zero
    private var length = CMTime.zero
    private var loadID = UUID()
    private var pendingSeek: Double?
    private var scrubbing = false
    private var transition: Task<Void, Never>?
    private var ticker: Task<Void, Never>?
    private var scheduledStart: CMTime?
    private var lastRecovery: Double?

    init() {
        for player in [original, result] {
            player.automaticallyWaitsToMinimizeStalling = false
            player.actionAtItemEnd = .pause
        }
        original.isMuted = true
    }

    func load(originalPath: String, resultPath: String) async {
        unload()
        let id = loadID
        do {
            async let restored = Clip.load(resultPath, failure: .result)
            async let source = Clip.load(originalPath, failure: .original)
            let output = try await restored
            let input = try await source
            guard loadID == id, !Task.isCancelled else { return }
            length = CMTimeMinimum(input.length, output.length)
            guard length.seconds > 0 else { throw Failure.timeline }
            originalStart = input.start
            resultStart = output.start
            for (player, clip) in [(original, input), (result, output)] {
                let item = AVPlayerItem(asset: clip.asset)
                item.forwardPlaybackEndTime = CMTimeAdd(clip.start, length)
                player.replaceCurrentItem(with: item)
            }
            let (originalReady, resultReady) = await seekPlayers(to: 0)
            guard loadID == id, !Task.isCancelled else { return }
            guard resultReady else { throw Failure.result }
            guard originalReady else { throw Failure.original }
            // Follow the result's clock, including its audio device when the
            // export has a soundtrack. The source contributes video only.
            if let timebase = result.currentItem?.timebase {
                original.sourceClock = CMTimebaseCopyUltimateSourceClock(timebase)
            }
            duration = length.seconds
            hasAudio = output.hasAudio
            size = output.size
        } catch {
            guard loadID == id, !Task.isCancelled else { return }
            unload()
            failure = error as? Failure ?? .timeline
        }
    }

    func unload() {
        loadID = UUID()
        holdPlayers()
        transition?.cancel()
        transition = nil
        for player in [original, result] {
            player.currentItem?.cancelPendingSeeks()
            player.replaceCurrentItem(with: nil)
        }
        original.sourceClock = nil
        pendingSeek = nil; scrubbing = false; lastRecovery = nil
        size = nil; duration = 0; position = 0; playing = false
        failure = nil; playbackInterrupted = false; hasAudio = false
    }

    func togglePlayback() {
        guard size != nil else { return }
        if playing {
            let stopAt = currentPosition()
            playing = false
            seek(to: stopAt)
        } else {
            playing = true; playbackInterrupted = false; lastRecovery = nil
            if position >= duration { seek(to: 0) }
            else { applyControls() }
        }
    }

    func setScrubbing(_ editing: Bool) {
        guard size != nil, editing != scrubbing else { return }
        scrubbing = editing
        if editing { seek(to: currentPosition()) }
        else { applyControls() }
    }

    func seek(to seconds: Double) {
        guard size != nil, seconds.isFinite else { return }
        position = min(duration, max(0, seconds))
        pendingSeek = position
        holdPlayers()
        applyControls()
    }

    func toggleAudio() {
        muted.toggle()
        result.isMuted = muted
    }

    private func holdPlayers() {
        if scheduledStart != nil {
            original.pause(); result.pause()
            scheduledStart = nil
        }
        original.cancelPendingPrerolls(); result.cancelPendingPrerolls()
        ticker?.cancel()
        ticker = nil
    }

    private func applyControls() {
        guard transition == nil, scheduledStart == nil else { return }
        let id = loadID
        transition = Task { [weak self] in await self?.applyControls(loadID: id) }
    }

    private func applyControls(loadID id: UUID) async {
        defer { if loadID == id { transition = nil } }
        while loadID == id, !Task.isCancelled {
            if let target = pendingSeek {
                pendingSeek = nil
                let (left, right) = await seekPlayers(to: target)
                guard loadID == id, !Task.isCancelled else { return }
                if pendingSeek != nil { continue }
                guard left && right else { interruptPlayback(); return }
            }
            guard playing, !scrubbing, position < duration else {
                if !scrubbing, position >= duration { playing = false }
                return
            }
            guard original.status == .readyToPlay, result.status == .readyToPlay else { interruptPlayback(); return }
            async let left = original.preroll(atRate: 1)
            async let right = result.preroll(atRate: 1)
            let (originalReady, resultReady) = await (left, right)
            guard loadID == id, !Task.isCancelled else { return }
            if pendingSeek != nil { continue }
            guard playing, !scrubbing else { return }
            guard originalReady && resultReady else { interruptPlayback(); return }
            let start = CMTimeAdd(CMClockGetTime(hostClock), CMTime(value: 1, timescale: 12))
            original.setRate(1, time: itemTime(position, from: originalStart), atHostTime: start)
            result.setRate(1, time: itemTime(position, from: resultStart), atHostTime: start)
            scheduledStart = start
            ticker = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(50))
                    guard !Task.isCancelled, let self else { return }
                    self.observePlayback()
                }
            }
            return
        }
    }

    private func seekPlayers(to seconds: Double) async -> (Bool, Bool) {
        async let left = original.seek(to: itemTime(seconds, from: originalStart), toleranceBefore: .zero, toleranceAfter: .zero)
        async let right = result.seek(to: itemTime(seconds, from: resultStart), toleranceBefore: .zero, toleranceAfter: .zero)
        return await (left, right)
    }

    private func itemTime(_ seconds: Double, from start: CMTime) -> CMTime {
        // Keep the last frame visible when scrubbing to the end of the range.
        let tick = CMTime(value: 1, timescale: 60_000)
        let offset = CMTimeMinimum(CMTime(seconds: seconds, preferredTimescale: tick.timescale),
                                   CMTimeMaximum(.zero, CMTimeSubtract(length, tick)))
        return CMTimeAdd(start, offset)
    }

    private func currentPosition() -> Double {
        guard let start = scheduledStart, CMClockGetTime(hostClock) >= start else { return position }
        let elapsed = CMTimeSubtract(result.currentTime(), resultStart).seconds
        return elapsed.isFinite ? min(duration, max(0, elapsed)) : position
    }

    private func observePlayback() {
        guard let start = scheduledStart else { return }
        let sinceStart = CMTimeSubtract(CMClockGetTime(hostClock), start).seconds
        guard sinceStart >= 0 else { return }
        let left = CMTimeSubtract(original.currentTime(), originalStart).seconds
        let right = CMTimeSubtract(result.currentTime(), resultStart).seconds
        guard left.isFinite, right.isFinite else { interruptPlayback(); return }
        if min(left, right) >= duration - 0.004 {
            playing = false
            seek(to: duration)
        } else if sinceStart > 0.2, (original.rate == 0 || result.rate == 0 || abs(left - right) > 0.035) {
            let commonTime = max(0, min(left, right))
            if let lastRecovery, abs(commonTime - lastRecovery) < 0.25 { interruptPlayback() }
            lastRecovery = commonTime
            seek(to: commonTime)
        } else {
            position = min(duration, max(0, right))
        }
    }

    private func interruptPlayback() {
        holdPlayers()
        playing = false
        playbackInterrupted = true
    }
}

/// Clip the two hardware-backed layers without copying decoded frames. Both
/// have the full picture geometry even when most of a layer is hidden.
private struct SplitVideoSurface: NSViewRepresentable {
    let original: AVPlayer
    let result: AVPlayer
    let fraction: CGFloat

    final class Surface: NSView {
        let left = AVPlayerLayer()
        let right = AVPlayerLayer()
        let leftCrop = CALayer()
        let rightCrop = CALayer()
        var fraction: CGFloat = 0.5

        init() {
            super.init(frame: .zero)
            wantsLayer = true
            layer?.backgroundColor = NSColor.black.cgColor
            for (clip, picture) in [(leftCrop, left), (rightCrop, right)] {
                clip.masksToBounds = true
                picture.videoGravity = .resize
                layer?.addSublayer(clip)
                clip.addSublayer(picture)
            }
        }

        required init?(coder: NSCoder) { nil }

        override func layout() {
            super.layout()
            updateCrops()
        }

        func updateCrops() {
            let size = bounds.size
            let divider = size.width * fraction
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            leftCrop.frame = CGRect(x: 0, y: 0, width: divider, height: size.height)
            rightCrop.frame = CGRect(x: divider, y: 0, width: size.width - divider, height: size.height)
            left.frame = CGRect(origin: .zero, size: size)
            right.frame = CGRect(origin: CGPoint(x: -divider, y: 0), size: size)
            CATransaction.commit()
        }
    }

    func makeNSView(context: Context) -> Surface {
        let surface = Surface()
        surface.left.player = original; surface.right.player = result
        surface.fraction = fraction
        return surface
    }

    func updateNSView(_ view: Surface, context: Context) {
        guard view.fraction != fraction else { return }
        view.fraction = fraction
        view.updateCrops()
    }

    static func dismantleNSView(_ view: Surface, coordinator: ()) {
        view.left.player = nil; view.right.player = nil
    }
}

struct VideoUpscaleComparison: View {
    let originalPath: String
    let resultPath: String
    @StateObject private var playback = VideoComparisonPlayback()
    @State private var split: CGFloat = 0.5
    @GestureState private var draggingTimeline = false
    @FocusState private var timelineFocused: Bool

    var body: some View {
        Group {
            if let size = playback.size {
                VStack(spacing: 8) {
                    GeometryReader { proxy in
                        let rect = AVMakeRect(aspectRatio: size, insideRect: CGRect(origin: .zero, size: proxy.size))
                        ZStack(alignment: .topLeading) {
                            Color.primary.opacity(0.025)
                            if rect.width > 0, rect.height > 0 {
                                SplitVideoSurface(original: playback.original, result: playback.result, fraction: split)
                                    .frame(width: rect.width, height: rect.height)
                                    .position(x: rect.midX, y: rect.midY)
                                    .allowsHitTesting(false).accessibilityHidden(true)
                                UpscaleComparisonDivider(visibleRect: rect, fraction: $split, originalTitle: "原视频")
                            }
                        }.coordinateSpace(name: UpscaleComparisonDivider.coordinateSpace)
                    }
                    controls
                }
            } else if let failure = playback.failure {
                if failure == .result {
                    ContentUnavailableView("无法读取超分视频", systemImage: "film")
                } else {
                    VStack(spacing: 8) {
                        VideoOutputPreview(path: resultPath)
                        Label(failure == .original ? "无法读取原视频" : "视频没有可对比的时间范围",
                              systemImage: "exclamationmark.circle")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            } else {
                ProgressView().controlSize(.small)
            }
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
            .task {
                split = 0.5
                await playback.load(originalPath: originalPath, resultPath: resultPath)
            }
            .onDisappear { playback.unload() }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                Button(action: playback.togglePlayback) {
                    Image(systemName: playback.playing ? "pause.fill" : "play.fill").frame(width: 19)
                }.help(playback.playing ? "暂停两侧" : "同步播放")
                    .accessibilityLabel(playback.playing ? "暂停两侧" : "同步播放")
                    .accessibilityIdentifier("videoComparisonPlayPause")
                timeline
                Text("\(time(playback.position)) / \(time(playback.duration))")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                if playback.hasAudio {
                    Button(action: playback.toggleAudio) {
                        Image(systemName: playback.muted ? "speaker.slash" : "speaker.wave.2").frame(width: 19)
                    }.help(playback.muted ? "恢复超分视频声音" : "静音")
                        .accessibilityLabel(playback.muted ? "恢复超分视频声音" : "静音")
                }
            }.controlSize(.small)
            if playback.playbackInterrupted {
                Text("同步播放已暂停，可再次播放。").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var timeline: some View {
        GeometryReader { proxy in
            Slider(value: Binding(get: { playback.position }, set: playback.seek), in: 0...playback.duration) {
                Text("两侧视频进度")
            }.labelsHidden().frame(maxWidth: .infinity, maxHeight: .infinity)
                .focused($timelineFocused)
                .accessibilityIdentifier("videoComparisonTimeline")
                .overlay {
                    // Use one drag across the track, including when the native
                    // slider sits inside the workspace's scrolling canvas.
                    Color.clear.contentShape(Rectangle())
                        .highPriorityGesture(DragGesture(minimumDistance: 0)
                            .updating($draggingTimeline) { _, tracking, _ in tracking = true }
                            .onChanged { drag in
                                playback.setScrubbing(true)
                                seekTimeline(drag.location.x, width: proxy.size.width)
                            }
                            .onEnded { drag in
                                seekTimeline(drag.location.x, width: proxy.size.width)
                                timelineFocused = true
                                playback.setScrubbing(false)
                            })
                        .accessibilityHidden(true)
                }
        }.frame(height: 19)
            .onChange(of: draggingTimeline) { _, dragging in
                if !dragging { playback.setScrubbing(false) }
            }
    }

    private func seekTimeline(_ x: CGFloat, width: CGFloat) {
        let inset = min(8, width / 2)
        let fraction = (x - inset) / max(1, width - inset * 2)
        playback.seek(to: Double(fraction) * playback.duration)
    }

    private func time(_ seconds: Double) -> String {
        let tenths = Int(min(seconds * 10, Double(Int32.max)))
        return String(format: "%d:%02d.%d", tenths / 600, tenths / 10 % 60, tenths % 10)
    }
}
