// Decode local generated media with the same macOS frameworks used by the App.
// Build: swiftc -O -parse-as-library ltx_av_media_probe.swift -o media-probe
import AVFoundation
import CoreVideo
import CryptoKit
import Foundation

enum ProbeError: Error { case invalid(String) }

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw ProbeError.invalid(message) }
}

func hex(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
}

func inspect(_ url: URL) async throws -> [String: Any] {
    let asset = AVURLAsset(url: url)
    let videos = try await asset.loadTracks(withMediaType: .video)
    let audios = try await asset.loadTracks(withMediaType: .audio)
    try require(videos.count == 1, "expected one video track")
    let video = videos[0]
    let rate = try await video.load(.nominalFrameRate)
    let size = try await video.load(.naturalSize)
    let range = try await video.load(.timeRange)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: video, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    try require(reader.startReading(), "start video reader: \(String(describing: reader.error))")
    var videoHash = SHA256(), timestamps: [Double] = []
    while let sample = output.copyNextSampleBuffer() {
        let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
        try require(pts.isFinite && (timestamps.last.map { pts > $0 } ?? true), "video timestamps")
        guard let pixels = CMSampleBufferGetImageBuffer(sample) else { throw ProbeError.invalid("missing pixels") }
        try require(CVPixelBufferLockBaseAddress(pixels, .readOnly) == kCVReturnSuccess, "lock pixels")
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixels) else { throw ProbeError.invalid("missing pixel storage") }
        let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
        try require(width == Int(size.width) && height == Int(size.height), "decoded video size changed")
        let stride = CVPixelBufferGetBytesPerRow(pixels)
        for row in 0..<height {
            videoHash.update(data: Data(bytes: base.advanced(by: row * stride), count: width * 4))
        }
        timestamps.append(pts)
    }
    try require(reader.status == .completed && !timestamps.isEmpty, "video reader: \(String(describing: reader.error))")
    var result: [String: Any] = [
        "file": url.lastPathComponent, "video_tracks": videos.count, "audio_tracks": audios.count,
        "width": Int(size.width), "height": Int(size.height), "fps": rate,
        "video_frames": timestamps.count, "video_duration_seconds": range.duration.seconds,
        "video_bgra_sha256": hex(videoHash.finalize()), "video_pts_seconds": timestamps
    ]
    if let audio = audios.first {
        try require(audios.count == 1, "expected at most one audio track")
        let audioReader = try AVAssetReader(asset: asset)
        let audioOutput = AVAssetReaderTrackOutput(track: audio, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false, AVLinearPCMIsNonInterleaved: false,
            AVLinearPCMIsBigEndianKey: false
        ])
        audioOutput.alwaysCopiesSampleData = false
        audioReader.add(audioOutput)
        try require(audioReader.startReading(), "start audio reader")
        var audioHash = SHA256(), frames = 0, samples = 0, sumSquares = 0.0, peak = 0.0
        var channels: UInt32 = 0, sampleRate = 0.0
        while let sample = audioOutput.copyNextSampleBuffer() {
            guard let description = CMSampleBufferGetFormatDescription(sample),
                  let format = CMAudioFormatDescriptionGetStreamBasicDescription(description),
                  let block = CMSampleBufferGetDataBuffer(sample) else { throw ProbeError.invalid("missing PCM") }
            if channels != 0 {
                try require(channels == format.pointee.mChannelsPerFrame && sampleRate == format.pointee.mSampleRate, "audio format changed")
            }
            channels = format.pointee.mChannelsPerFrame
            sampleRate = format.pointee.mSampleRate
            let count = CMBlockBufferGetDataLength(block)
            try require(count % 2 == 0, "unaligned PCM length")
            var data = Data(count: count)
            let status = data.withUnsafeMutableBytes { bytes in
                CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: count, destination: bytes.baseAddress!)
            }
            try require(status == kCMBlockBufferNoErr, "read PCM bytes")
            audioHash.update(data: data)
            data.withUnsafeBytes { bytes in
                for index in stride(from: 0, to: count, by: 2) {
                    let value = Double(bytes.loadUnaligned(fromByteOffset: index, as: Int16.self).littleEndian) / 32768
                    sumSquares += value * value
                    peak = max(peak, abs(value))
                    samples += 1
                }
            }
            frames += CMSampleBufferGetNumSamples(sample)
        }
        try require(audioReader.status == .completed && samples > 0 && sampleRate > 0, "audio reader: \(String(describing: audioReader.error))")
        result.merge([
            "audio_channels": channels, "audio_sample_rate": sampleRate, "audio_frames": frames,
            "audio_duration_seconds": Double(frames) / sampleRate,
            "audio_pcm_s16le_sha256": hex(audioHash.finalize()),
            "audio_rms": sqrt(sumSquares / Double(samples)), "audio_peak": peak
        ]) { _, new in new }
    }
    return result
}

@main struct MediaProbe {
    static func main() async {
        do {
            try require(CommandLine.arguments.count > 1, "usage: media-probe VIDEO.mp4 [VIDEO.mp4 ...]")
            var results: [[String: Any]] = []
            for path in CommandLine.arguments.dropFirst() {
                results.append(try await inspect(URL(fileURLWithPath: path)))
            }
            let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data([10]))
        } catch {
            FileHandle.standardError.write(Data("media verification failed: \(error)\n".utf8))
            exit(1)
        }
    }
}
