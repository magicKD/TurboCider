import Foundation
import AVFoundation
import CryptoKit

@main struct LTXWorkerTests {
    static func require(_ value: Bool, _ message: String) throws {
        guard value else { throw NativeFailure(message: message) }
    }

    static func video(_ url: URL) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 64, AVVideoHeightKey: 64])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
            sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                                         kCVPixelBufferWidthKey as String: 64, kCVPixelBufferHeightKey as String: 64])
        writer.add(input)
        try require(writer.startWriting(), "Cannot start fixture encoder")
        writer.startSession(atSourceTime: .zero)
        var buffer: CVPixelBuffer?
        try require(CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32BGRA,
                                       nil, &buffer) == kCVReturnSuccess, "Cannot allocate fixture frame")
        CVPixelBufferLockBaseAddress(buffer!, [])
        memset(CVPixelBufferGetBaseAddress(buffer!), 127, CVPixelBufferGetDataSize(buffer!))
        CVPixelBufferUnlockBaseAddress(buffer!, [])
        for frame in 0..<9 {
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(for: .milliseconds(1))
                try require(writer.status == .writing, "Fixture writer failed")
            }
            try require(adaptor.append(buffer!, withPresentationTime: CMTime(value: Int64(frame), timescale: 24)),
                        "Cannot append fixture frame")
        }
        input.markAsFinished()
        await writer.finishWriting()
        try require(writer.status == .completed, "Fixture encoding failed")
    }

    static func main() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("tc-worker-tests-\(UUID())")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let fixture = root.appendingPathComponent("fixture.mp4")
        try await video(fixture)
        let output = root.appendingPathComponent("published.mp4")
        let original = Data("existing product must survive failure".utf8)
        let executable = root.appendingPathComponent("worker")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let fixtureLiteral = String(decoding: try encoder.encode(fixture.path), as: UTF8.self)
        let finalLiteral = String(decoding: try encoder.encode(output.path), as: UTF8.self)
        func worker(_ mode: String) throws {
            let script = """
            #!/usr/bin/python3
            import json, os, shutil, sys, time
            mode = '\(mode)'
            request = json.load(open(sys.argv[3]))
            v2 = request.get('schema_version') == 2
            out = request['outputs'][0] if v2 else request
            path = out['path'] if v2 else out['output']
            sampling = request['sampling'] if v2 else request
            if mode == 'cancel': time.sleep(60)
            if mode == 'huge_result': sys.stdout.write('x'*(9<<20)); sys.exit(0)
            if mode == 'huge_events': sys.stderr.write('x'*(17<<20)); sys.exit(1)
            if mode == 'empty_json': print('{}'); sys.exit(0)
            if mode == 'empty_file': open(path, 'wb').close()
            elif mode == 'invalid_media': open(path, 'wb').write(b'not a video')
            elif mode == 'symlink': os.symlink(\(fixtureLiteral), path)
            elif mode != 'missing_file': shutil.copyfile(\(fixtureLiteral), path)
            result = dict(schema_version=1, model=request['model'], operation=request['operation'],
                          output=path, width=out['width'], height=out['height'],
                          steps=sampling['steps'], seed=sampling['seed'], warmup=False)
            if mode == 'old_path': result['output'] = \(finalLiteral)
            if mode == 'wrong_seed': result['seed'] += 1
            if v2:
                selector = request['execution']['streaming']
                summary = dict(schema_version=1, target_request_memory_bytes=selector['target_request_memory_bytes'],
                    calibrated_request_bytes=1<<30, preset_id='fixture', preset_revision=1, catalog_revision='fixture-v1',
                    component_policy_revision='fixture-components', execution_container='embedded_app', memory_scope='request',
                    receipt_schema_version=3, receipt_source_generation=1, receipt_verifier_revision='fixture-v1', actual_plan_verified=True)
                for key in ['record_digest', 'resolution_digest', 'source_digest', 'workload_digest', 'runtime_digest',
                            'device_digest', 'authorized_layout_digest', 'actual_layout_digest', 'receipt_digest']:
                    summary[key] = 'a'*64
                if mode == 'false_verified': summary['actual_plan_verified'] = False
                if mode == 'numeric_verified': summary['actual_plan_verified'] = 1
                if mode == 'wrong_target': summary['target_request_memory_bytes'] += 1
                if mode == 'wrong_layout': summary['actual_layout_digest'] = 'b'*64
                if mode != 'missing_summary': result['public_streaming'] = summary
            print(json.dumps(result))
            """
            try script.write(to: executable, atomically: true, encoding: .utf8)
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
        }
        var request = NativeRequest(prompt: "fixture", output: output.path)
        request.model = "ltx-2.5-distilled"; request.operation = "video.generate"
        request.width = 64; request.height = 64; request.frames = 9; request.fps = 24
        request.steps = 1; request.seed = 42; request.audio = false
        request.residency = "component_staged"
        let intent = NativeRequestV2(legacy: request, targetBytes: 8 << 30)
        for mode in ["empty_json", "old_path", "wrong_seed", "empty_file", "invalid_media", "missing_file", "symlink",
                     "false_verified", "numeric_verified", "wrong_target", "wrong_layout", "missing_summary",
                     "huge_result", "huge_events"] {
            try original.write(to: output)
            try worker(mode)
            var failed = false
            do {
                _ = try await LTXWorker.generate(model: root, request: intent, outputPath: output.path,
                    executable: executable, onEvent: { _ in })
            } catch {
                failed = true
                if ["empty_json", "old_path", "wrong_seed", "false_verified", "numeric_verified",
                    "wrong_target", "wrong_layout", "missing_summary"].contains(mode) {
                    try require(error.localizedDescription.contains("worker_result_"),
                                "Wrong failure for result case \(mode): \(error)")
                }
            }
            try require(failed, "Worker accepted invalid case: \(mode)")
            try require(try Data(contentsOf: output) == original, "Failure replaced existing output: \(mode)")
        }
        try worker("valid")
        var wrongDimensions = intent
        wrongDimensions.outputs[0].width = 128
        var rejectedDimensions = false
        do {
            _ = try await LTXWorker.generate(model: root, request: wrongDimensions, outputPath: output.path,
                executable: executable, onEvent: { _ in })
        } catch { rejectedDimensions = true }
        try require(rejectedDimensions && (try Data(contentsOf: output)) == original, "Video geometry was not verified")
        for field in ["frames", "audio", "preset"] {
            var mismatch = intent
            if field == "frames" { mismatch.outputs[0].frames = 8 }
            if field == "audio" { mismatch.outputs[0].audio = true }
            if field == "preset" {
                mismatch.execution.streaming?.selection = "preset"
                mismatch.execution.streaming?.preset_id = "wrong-preset"
                mismatch.execution.streaming?.preset_revision = 1
            }
            var rejected = false
            do {
                _ = try await LTXWorker.generate(model: root, request: mismatch, outputPath: output.path,
                    executable: executable, onEvent: { _ in })
            } catch { rejected = true }
            try require(rejected && (try Data(contentsOf: output)) == original,
                        "Worker did not reject mismatched \(field)")
        }
        let blockedOutput = root.appendingPathComponent("blocked.mp4", isDirectory: true)
        try fm.createDirectory(at: blockedOutput, withIntermediateDirectories: true)
        let marker = blockedOutput.appendingPathComponent("keep")
        try original.write(to: marker)
        var blocked = request
        blocked.output = blockedOutput.path
        var failedPublication = false
        do {
            _ = try await LTXWorker.generate(model: root, request: blocked,
                executable: executable, onEvent: { _ in })
        } catch { failedPublication = error.localizedDescription.contains("worker_publish_failed") }
        try require(failedPublication && (try Data(contentsOf: marker)) == original,
                    "Failed publication must preserve the existing destination")
        let result = try await LTXWorker.generate(model: root, request: intent, outputPath: output.path,
            executable: executable, onEvent: { _ in })
        let published = try JSONSerialization.jsonObject(with: result) as! [String: Any]
        let receipt = published["worker_artifact"] as! [String: Any]
        let data = try Data(contentsOf: output)
        let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try require(published["output"] as? String == output.path && receipt["sha256"] as? String == sha,
                    "Publication receipt does not identify the published bytes")
        try require(data == Data(contentsOf: fixture), "Atomic publication changed video bytes")
        try original.write(to: output)
        _ = try await LTXWorker.generate(model: root, request: request, executable: executable, onEvent: { _ in })
        try require(try Data(contentsOf: output) == data, "Legacy worker did not publish a verified video")
        try original.write(to: output)
        try worker("cancel")
        let cancellationRequest = request
        let task = Task {
            try await LTXWorker.generate(model: root, request: cancellationRequest, executable: executable, onEvent: { _ in })
        }
        try await Task.sleep(for: .milliseconds(300))
        task.cancel()
        do { _ = try await task.value; throw NativeFailure(message: "Cancellation was accepted as success") }
        catch is CancellationError {}
        try require(try Data(contentsOf: output) == original, "Cancellation replaced existing output")
        try require(try fm.contentsOfDirectory(atPath: root.path).allSatisfy { !$0.hasPrefix(".tc-ltx-staging-") },
                    "Worker leaked a staging directory")
        print("PASS LTX worker: strict result correlation, verified summary, full video decode, SHA receipt, atomic publication, rollback and cancellation")
    }
}
