import Foundation
import Darwin
import AVFoundation
import CryptoKit

/// The CLI owns the disposable denoiser process and execs the clean VAE
/// finalizer. Never enable EXEC_FINALIZER in the desktop process itself.
enum LTXWorker {
    private final class ProcessExit: @unchecked Sendable {
        private let lock = NSLock()
        private var finished = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        func signal() {
            lock.lock()
            finished = true
            let pending = waiters
            waiters.removeAll()
            lock.unlock()
            pending.forEach { $0.resume() }
        }
        func wait() async {
            await withCheckedContinuation { continuation in
                lock.lock()
                if finished { lock.unlock(); continuation.resume() }
                else { waiters.append(continuation); lock.unlock() }
            }
        }
    }
    static func accepts(_ request: NativeRequest) -> Bool {
        request.model == "ltx-2.5-distilled" && request.residency == "component_staged"
    }

    static func generate(model: URL, request: NativeRequest, executable: URL? = nil,
                         onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await generatePayload(model: model, request: request,
                                  expected: NativeRequestV2(legacy: request),
                                  outputPath: request.output,
                                  executable: executable, onEvent: onEvent)
    }

    /// Public streaming intent is resolved inside the disposable worker. The
    /// desktop process never opens the model, creates a SourceLease, or passes
    /// an authority/fd/pointer across the process boundary.
    static func generate(model: URL, request: NativeRequestV2, outputPath: String,
                         executable: URL? = nil,
                         onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        try await generatePayload(model: model, request: request,
                                  expected: request,
                                  outputPath: outputPath,
                                  executable: executable, onEvent: onEvent)
    }

    private static func generatePayload<Request: Encodable & Sendable>(
            model: URL, request: Request, expected: NativeRequestV2, outputPath: String,
            executable: URL?,
            onEvent: @escaping @Sendable (NativeEvent) -> Void) async throws -> Data {
        let worker = Task.detached(priority: .userInitiated) { () throws -> Data in
            let fm = FileManager.default
            let directory = fm.temporaryDirectory.appendingPathComponent("tc-ltx-app-\(UUID())")
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: directory) }
            let destination = URL(fileURLWithPath: outputPath).standardizedFileURL
            guard expected.outputs.count == 1, expected.outputs[0].path == outputPath,
                  expected.model == "ltx-2.5-distilled", destination.pathExtension == "mp4" else {
                throw NativeFailure(message: "worker_request_invalid: LTX 输出请求不一致。")
            }
            let staging = destination.deletingLastPathComponent()
                .appendingPathComponent(".tc-ltx-staging-\(UUID())", isDirectory: true)
            try fm.createDirectory(at: staging, withIntermediateDirectories: true,
                                   attributes: [.posixPermissions: 0o700])
            defer { try? fm.removeItem(at: staging) }
            let artifact = staging.appendingPathComponent("output.mp4")
            var bound = expected
            bound.outputs[0].path = artifact.path
            var payload = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
            if payload["schema_version"] as? Int == 2 {
                var outputs = payload["outputs"] as! [[String: Any]]
                outputs[0]["path"] = artifact.path
                payload["outputs"] = outputs
            } else {
                payload["output"] = artifact.path
            }
            let input = directory.appendingPathComponent("request.json")
            try JSONSerialization.data(withJSONObject: payload).write(to: input, options: .atomic)
            let output = directory.appendingPathComponent("result.json")
            let events = directory.appendingPathComponent("events.jsonl")
            fm.createFile(atPath: output.path, contents: nil)
            fm.createFile(atPath: events.path, contents: nil)
            let stdout = try FileHandle(forWritingTo: output)
            let stderr = try FileHandle(forWritingTo: events)
            let reader = try FileHandle(forReadingFrom: events)
            defer { try? stdout.close(); try? stderr.close(); try? reader.close() }
            let child = Process()
            child.executableURL = executable ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("turbocider")
            child.arguments = ["generate", model.path, input.path]
            child.standardOutput = stdout; child.standardError = stderr
            var environment = ProcessInfo.processInfo.environment
            environment["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = environment["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] ?? TensorCache.sharedRoot.path
            child.environment = environment
            let exit = ProcessExit()
            child.terminationHandler = { _ in exit.signal() }
            try Task.checkCancellation()
            try child.run()
            do {
                var pending = Data(), failure = "LTX 视频生成失败。"
                var cancelledAt: ContinuousClock.Instant?
                func checkLogSizes() throws {
                    for (url, limit) in [(output, 8 << 20), (events, 16 << 20)] {
                        let size = (try fm.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.uint64Value ?? .max
                        guard size <= UInt64(limit) else {
                            throw NativeFailure(message: "worker_log_limit: 视频进程日志或结果超过大小限制。")
                        }
                    }
                }
                func drain(limit: Int = 1 << 20) throws {
                    var consumed = 0
                    while consumed < limit, let data = try reader.read(upToCount: 256 * 1024), !data.isEmpty {
                        consumed += data.count
                        pending.append(data)
                        while let end = pending.firstIndex(of: 0x0a) {
                            let line = Data(pending[..<end]); pending.removeSubrange(...end)
                            if let event = try? JSONDecoder().decode(NativeEvent.self, from: line) { onEvent(event) }
                            else if !line.isEmpty { failure = String(decoding: line.suffix(4000), as: UTF8.self) }
                        }
                        if pending.count > 64 * 1024 { pending = Data(pending.suffix(64 * 1024)) }
                    }
                }
                while child.isRunning {
                    try checkLogSizes()
                    try drain()
                    if Task.isCancelled, cancelledAt == nil { cancelledAt = .now; child.terminate() }
                    if let cancelledAt, cancelledAt.duration(to: .now) > .seconds(5) {
                        kill(child.processIdentifier, SIGKILL)
                    }
                    // Preserve the cancellation grace period even after task cancellation.
                    await Task.detached { try? await Task.sleep(for: .milliseconds(100)) }.value
                }
                await exit.wait()
                try checkLogSizes()
                try drain(limit: 16 << 20)
                if Task.isCancelled || child.terminationStatus == 2 { throw CancellationError() }
                if !pending.isEmpty { failure = String(decoding: pending.suffix(4000), as: UTF8.self) }
                guard child.terminationStatus == 0 else { throw NativeFailure(message: failure) }
                let attributes = try fm.attributesOfItem(atPath: output.path)
                guard ((attributes[.size] as? NSNumber)?.uint64Value ?? .max) <= (8 << 20) else {
                    throw NativeFailure(message: "worker_result_invalid: 返回结果过大。")
                }
                let result = try Data(contentsOf: output)
                try validateResult(result, request: bound)
                let identity = try artifactIdentity(artifact)
                try await validateVideo(artifact, request: bound)
                var hash = SHA256()
                let bytes = try FileHandle(forReadingFrom: artifact)
                defer { try? bytes.close() }
                guard try artifactIdentity(artifact, fd: bytes.fileDescriptor) == identity else {
                    throw NativeFailure(message: "worker_artifact_changed: 视频在验证期间发生变化。")
                }
                var remaining = UInt64(identity[2])
                while remaining > 0 {
                    try Task.checkCancellation()
                    guard let chunk = try bytes.read(upToCount: Int(min(remaining, 1024 * 1024))), !chunk.isEmpty else {
                        throw NativeFailure(message: "worker_artifact_changed: 视频在验证期间被截断。")
                    }
                    hash.update(data: chunk)
                    remaining -= UInt64(chunk.count)
                }
                let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
                let artifactBytes = try fm.attributesOfItem(atPath: artifact.path)[.size] as! NSNumber
                var published = try JSONSerialization.jsonObject(with: result) as! [String: Any]
                published["output"] = destination.path
                published["worker_artifact"] = ["schema_version": 1, "staged_output": artifact.path,
                    "published_output": destination.path, "bytes": artifactBytes, "sha256": digest,
                    "media_verified": true] as [String: Any]
                let receipt = try JSONSerialization.data(withJSONObject: published, options: [.sortedKeys])
                try Task.checkCancellation()
                guard try artifactIdentity(artifact, fd: bytes.fileDescriptor) == identity,
                      try artifactIdentity(artifact) == identity else {
                    throw NativeFailure(message: "worker_artifact_changed: 视频在验证期间发生变化。")
                }
                guard rename(artifact.path, destination.path) == 0 else {
                    throw NativeFailure(message: "worker_publish_failed: 无法发布已验证的视频。")
                }
                return receipt
            } catch {
                if child.isRunning { kill(child.processIdentifier, SIGKILL) }
                await exit.wait()
                throw error
            }
        }
        return try await withTaskCancellationHandler { try await worker.value } onCancel: { worker.cancel() }
    }

    private static func artifactIdentity(_ url: URL, fd: Int32? = nil) throws -> [Int64] {
        var value = stat()
        let status = fd.map { fstat($0, &value) } ?? lstat(url.path, &value)
        guard status == 0, value.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), value.st_size > 0 else {
            throw NativeFailure(message: "worker_artifact_invalid: 视频文件为空或不是普通文件。")
        }
        return [Int64(value.st_dev), Int64(bitPattern: value.st_ino), value.st_size,
                Int64(value.st_mtimespec.tv_sec), Int64(value.st_mtimespec.tv_nsec),
                Int64(value.st_ctimespec.tv_sec), Int64(value.st_ctimespec.tv_nsec)]
    }

    static func validateResult(_ data: Data, request: NativeRequestV2) throws {
        struct Result: Decodable {
            let schema_version: Int
            let model: String
            let operation: String
            let output: String
            let width: Int
            let height: Int
            let steps: Int
            let seed: Int
            let warmup: Bool
            let public_streaming: NativeStreamingResultSummary?
        }
        let result: Result
        do { result = try JSONDecoder().decode(Result.self, from: data) }
        catch { throw NativeFailure(message: "worker_result_invalid: 视频进程未返回有效结果。") }
        guard request.outputs.count == 1, let output = request.outputs.first,
              result.schema_version == 1, !result.warmup,
              result.model == request.model, result.operation == request.operation,
              result.output == output.path, result.width == output.width,
              result.height == output.height, result.steps == request.sampling.steps,
              result.seed == request.sampling.seed else {
            throw NativeFailure(message: "worker_result_mismatch: 视频结果与本次请求不一致。")
        }
        guard let selector = request.execution.streaming else { return }
        guard let actual = result.public_streaming else {
            throw NativeFailure(message: "worker_result_invalid: 缺少流式执行证明。")
        }
        let digests = [actual.record_digest, actual.resolution_digest, actual.source_digest,
            actual.workload_digest, actual.runtime_digest, actual.device_digest,
            actual.authorized_layout_digest, actual.actual_layout_digest, actual.receipt_digest]
        guard actual.schema_version == 1, actual.actual_plan_verified,
              actual.target_request_memory_bytes == selector.target_request_memory_bytes,
              actual.calibrated_request_bytes > 0,
              actual.calibrated_request_bytes <= selector.target_request_memory_bytes,
              actual.authorized_layout_digest == actual.actual_layout_digest,
              !actual.preset_id.isEmpty, actual.preset_revision > 0, !actual.catalog_revision.isEmpty,
              !actual.component_policy_revision.isEmpty, !actual.execution_container.isEmpty,
              !actual.memory_scope.isEmpty, !actual.receipt_verifier_revision.isEmpty,
              (2...3).contains(actual.receipt_schema_version), actual.receipt_source_generation > 0,
              digests.allSatisfy({ $0.count == 64 && $0.allSatisfy { "0123456789abcdef".contains($0) } }),
              selector.preset_id.map({ $0 == actual.preset_id }) ?? true,
              selector.preset_revision.map({ $0 == actual.preset_revision }) ?? true,
              selector.catalog_revision.map({ $0 == actual.catalog_revision }) ?? true,
              selector.expected_resolution_digest.map({ $0 == actual.resolution_digest }) ?? true else {
            throw NativeFailure(message: "worker_result_mismatch: 流式执行证明与请求不一致。")
        }
    }

    private static func validateVideo(_ url: URL, request: NativeRequestV2) async throws {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              (attributes[.size] as? NSNumber)?.uint64Value ?? 0 > 0 else {
            throw NativeFailure(message: "worker_artifact_invalid: 视频文件为空或不是普通文件。")
        }
        let expected = request.outputs[0]
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NativeFailure(message: "worker_artifact_invalid: 缺少视频轨道。")
        }
        let size = try await track.load(.naturalSize)
        let duration = try await track.load(.timeRange).duration
        guard Int(size.width) == expected.width, Int(size.height) == expected.height,
              expected.frames > 0, expected.fps > 0, duration.seconds.isFinite,
              abs(duration.seconds - Double(expected.frames) / Double(expected.fps)) <= 1.0 / Double(expected.fps) else {
            throw NativeFailure(message: "worker_artifact_invalid: 视频尺寸或时长与请求不一致。")
        }
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let hasAudio = !audioTracks.isEmpty
        if hasAudio != expected.audio {
            throw NativeFailure(message: "worker_artifact_invalid: 音频轨道与请求不一致。")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        guard reader.canAdd(output) else { throw NativeFailure(message: "worker_artifact_invalid: 无法解码视频。") }
        reader.add(output)
        guard reader.startReading() else { throw NativeFailure(message: "worker_artifact_invalid: 视频解码失败。") }
        defer { reader.cancelReading() }
        var frames = 0
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard CMSampleBufferGetImageBuffer(sample) != nil, frames < expected.frames else {
                throw NativeFailure(message: "worker_artifact_invalid: 视频帧无效。")
            }
            frames += 1
        }
        guard reader.status == .completed, frames == expected.frames else {
            throw NativeFailure(message: "worker_artifact_invalid: 视频不完整或无法解码。")
        }
    }
}
