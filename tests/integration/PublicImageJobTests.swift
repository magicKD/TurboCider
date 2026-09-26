import Foundation
import CoreGraphics
import ImageIO

/// Fake worker tests cover App transactions/recovery, not native qualification.
enum PublicImageJobTests {
    @MainActor static func run(root: URL) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        func check(_ value: Bool, _ message: String) throws { if !value { throw NativeFailure(message: message) } }
        let pngURL = root.appendingPathComponent("fixture.png")
        let context = CGContext(data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.3, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let writer = CGImageDestinationCreateWithURL(pngURL as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(writer, context.makeImage()!, nil)
        try check(CGImageDestinationFinalize(writer), "fixture encode failed")
        let png = try Data(contentsOf: pngURL)
        func worker(_ mode: String) throws -> URL {
            let path = root.appendingPathComponent("worker-\(mode).py")
            let code = """
            #!/usr/bin/python3
            import sys,json,base64,hashlib,signal,time,uuid
            from pathlib import Path
            if sys.stdin.buffer.read(1)!=b'\\1':sys.exit(1)
            wire=json.loads(Path(sys.argv[2]).read_text());request=wire['native_request_v2'];out=request['outputs'][0]
            mode='\(mode)'
            if mode=='hang':signal.signal(signal.SIGTERM,signal.SIG_IGN)
            pixels=base64.b64decode('\(png.base64EncodedString())') if mode!='invalid_png' else b'not a PNG'
            Path(out['path']).write_bytes(pixels)
            if mode=='hang':
                while True:time.sleep(1)
            selector=request['execution']['streaming'];budget=selector['target_request_memory_bytes']
            exact=dict(selector,selection='preset',preset_id='fixture',preset_revision=1,catalog_revision='catalog',expected_resolution_digest='resolution')
            selection=dict(preset_id='fixture',preset_revision=1,record_digest='record',release_channel='public',target_request_memory_bytes=budget,calibrated_request_bytes=1024,memory_scope='request',layout_digest='layout',component_policy_revision='policy',execution_container='cli_worker')
            identity=dict(source_digest='source',runtime_digest='runtime',device_digest='device')
            resolution=dict(schema_version=1,status='resolved',request_digest='workload',resolution_digest='resolution',catalog_revision='catalog',requested_selector=selector,exact_selector=exact,selection=selection,identity=identity)
            if mode in ['events','bad_event','partial_event']:
                event=dict(event_schema_version=1,kind='resolved',sequence=1,job_id=wire['job_id'],request_id=wire['request_id'],request_digest=wire['request_digest'],execution_container='cli_worker',runtime_fingerprint='\(NativeEngine.runtimeBuildIdentity())',payload=resolution)
                print('TC_EVENT\\t'+json.dumps(event),file=sys.stderr,flush=True)
                event.update(kind='progress',sequence=2,payload=dict(sequence=10,phase='denoise',completed=0,total=4,elapsed_seconds=1))
                if mode=='bad_event':event['request_id']=str(uuid.uuid4())
                print('TC_EVENT\\t'+json.dumps(event),file=sys.stderr,flush=True)
                event.update(sequence=3,payload=dict(sequence=11,phase='denoise',completed=2,total=4,elapsed_seconds=3))
                print('TC_EVENT\\t'+json.dumps(event),file=sys.stderr,flush=True)
                if mode=='partial_event':sys.stderr.write('TC_EVENT\\t{');sys.stderr.flush()
                time.sleep(1)
            summary=dict(selection,**identity,schema_version=1,catalog_revision='catalog',resolution_digest='resolution',workload_digest='workload',authorized_layout_digest='layout',actual_layout_digest='layout',receipt_schema_version=3,receipt_source_generation=1,receipt_digest='receipt',receipt_verifier_revision='verifier',actual_plan_verified=True)
            result=dict(schema_version=1,model=request['model'],operation=request['operation'],output=out['path'],width=out['width'],height=out['height'],steps=request['sampling']['steps'],seed=request['sampling']['seed'],warmup=False,public_streaming=summary)
            artifact=dict(path=out['path'],size=len(pixels),sha256=hashlib.sha256(pixels).hexdigest() if mode!='bad_hash' else '0'*64)
            terminal=dict(protocol_version=1,job_id=wire['job_id'],request_id=wire['request_id'] if mode!='bad_id' else str(uuid.uuid4()),request_digest=wire['request_digest'],status='succeeded',actual_container='cli_worker',runtime_fingerprint='\(NativeEngine.runtimeBuildIdentity())',resolution_digest='resolution',record_digest='record',layout_digest='layout',error=None,resolution=resolution,result=result,public_streaming_summary=summary,artifact=artifact)
            print(json.dumps(terminal))
            """
            try Data(code.utf8).write(to: path)
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: path.path)
            return path
        }
        func request(_ store: NativeJobStore) -> NativeRequest {
            var value = NativeRequest(prompt: "fixture", output: store.directory.appendingPathComponent("outputs/final.png").path)
            value.model = "z-image-turbo"; value.operation = "image.generate"; value.width = 64; value.height = 64; value.steps = 4
            return value
        }
        for mode in ["success", "bad_id", "bad_hash", "invalid_png", "bad_event", "partial_event"] {
            let store = NativeJobStore(directory: root.appendingPathComponent(mode), workerExecutable: try worker(mode))
            let original = request(store)
            var failure: Error?
            do { _ = try await store.generate(modelURL: root, request: original, streamingRequest: NativeRequestV2(legacy: original, targetBytes: 10 << 30)) }
            catch { failure = error }
            try check(!store.busy && !store.workerCleanupPending, "Worker did not release App busy: \(mode)")
            try check(store.jobs.count == 1 && store.jobs[0].publicWorker?.exitConfirmed == true, "Missing durable worker exit: \(mode)")
            try check(store.jobs[0].state == (mode == "success" ? "succeeded" : "failed"), "Wrong terminal state: \(mode), \(String(describing: failure))")
            if mode == "success" {
                try check(failure == nil, "Success threw")
                let actual = try Data(contentsOf: URL(fileURLWithPath: original.output))
                try check(actual == png, "Published pixels changed")
                let reopened = NativeJobStore(directory: store.directory)
                try check(reopened.jobs[0].state == "succeeded" && !reopened.busy, "Successful worker history did not restore")
            } else { try check(!fm.fileExists(atPath: original.output), "Rejected worker published a file") }
            let leftovers = try fm.contentsOfDirectory(atPath: store.directory.appendingPathComponent("outputs").path)
            try check(!leftovers.contains(where: { $0.hasPrefix(".tc-image-staging-") }), "Worker staging leaked: \(mode)")
        }
        let streaming = NativeJobStore(directory: root.appendingPathComponent("events"), workerExecutable: try worker("events"))
        let streamedRequest = request(streaming)
        let streamed = Task { try await streaming.generate(modelURL: root, request: streamedRequest,
            streamingRequest: NativeRequestV2(legacy: streamedRequest, targetBytes: 10 << 30)) }
        var observed = false
        for _ in 0..<100 {
            if let job = streaming.jobs.first, job.phase == "denoise", job.completed == 2,
               job.elapsed == 3, job.state == "running", job.publicWorker?.exitConfirmed == false {
                observed = true; break
            }
            try await Task.sleep(for: .milliseconds(25))
        }
        _ = try await streamed.value
        try check(observed, "Progress was not visible before worker exit")
        try check(streaming.jobs[0].state == "succeeded", "Events prevented valid publication")
        let reference = streaming.jobs[0].publicWorker!
        let diagnostics = reference.directory(jobID: streaming.jobs[0].id, store: streaming.directory)
        let wire = try JSONSerialization.jsonObject(with: Data(contentsOf: diagnostics.appendingPathComponent("input.json"))) as! [String: Any]
        let intent = try JSONDecoder().decode(NativeRequestV2.self, from: JSONSerialization.data(withJSONObject: wire["native_request_v2"]!))
        let log = try Data(contentsOf: diagnostics.appendingPathComponent("stderr.log"))
        let records = log.split(separator: 10).filter { $0.starts(with: Data("TC_EVENT\t".utf8)) }.map { Data($0) }
        try check(records.count == 3, "Missing saved telemetry")
        func parser() -> WorkerEventStream {
            WorkerEventStream(jobID: streaming.jobs[0].id, reference: reference, request: intent) { _ in }
        }
        let fragmented = parser()
        try fragmented.consume(Data(repeating: 120, count: 200_000)) // Bounded non-event diagnostic discard.
        try fragmented.consume(Data([10]))
        for byte in log { try fragmented.consume(Data([byte])) }
        try fragmented.finish(resolution: nil)
        let firstObject = try JSONSerialization.jsonObject(with: records[0].dropFirst(9)) as! [String: Any]
        var different = firstObject["payload"] as! [String: Any]
        different["resolution_digest"] = "different-terminal"
        let differentResolution = try JSONDecoder().decode(NativeStreamingResolution.self, from: JSONSerialization.data(withJSONObject: different))
        do { try fragmented.finish(resolution: differentResolution); throw NativeFailure(message: "Accepted mismatched terminal resolution") }
        catch let error as NativeFailure { try check(!error.message.hasPrefix("Accepted"), error.message) }
        for key in ["job_id", "request_id", "request_digest", "execution_container", "runtime_fingerprint", "event_schema_version", "sequence", "kind"] {
            var value = try JSONSerialization.jsonObject(with: records[0].dropFirst(9)) as! [String: Any]
            value[key] = ["event_schema_version", "sequence"].contains(key) ? 0 : "wrong"
            var bad = Data("TC_EVENT\t".utf8); bad.append(try JSONSerialization.data(withJSONObject: value)); bad.append(10)
            do { try parser().consume(bad); throw NativeFailure(message: "Accepted bad event \(key)") }
            catch let error as NativeFailure { try check(!error.message.hasPrefix("Accepted"), error.message) }
            catch {} // Typed decoder rejection is also valid.
        }
        let duplicate = parser()
        var first = records[0]; first.append(10)
        try duplicate.consume(first)
        do { try duplicate.consume(first); throw NativeFailure(message: "Accepted duplicate event") }
        catch let error as NativeFailure { try check(!error.message.hasPrefix("Accepted"), error.message) }
        let truncated = parser()
        try truncated.consume(records[0])
        do { try truncated.finish(resolution: nil); throw NativeFailure(message: "Accepted truncated event") }
        catch let error as NativeFailure { try check(!error.message.hasPrefix("Accepted"), error.message) }
        var oversized = Data("TC_EVENT\t".utf8); oversized.append(Data(repeating: 120, count: 128 * 1024 + 1))
        do { try parser().consume(oversized); throw NativeFailure(message: "Accepted oversized event") }
        catch let error as NativeFailure { try check(!error.message.hasPrefix("Accepted"), error.message) }
        print("PASS: live worker telemetry before exit, fragmented framing and identity/order/size checks; malformed/truncated events block publication")
        let live = NativeJobStore(directory: root.appendingPathComponent("live"), workerExecutable: try worker("hang"))
        let original = request(live)
        let task = Task { try await live.generate(modelURL: root, request: original, streamingRequest: NativeRequestV2(legacy: original, targetBytes: 10 << 30)) }
        var restored: NativeJobStore?
        var recoveryFailure: Error?
        do {
            var ready = false
            for _ in 0..<400 {
                if let reference = live.jobs.first?.publicWorker, fm.fileExists(atPath: reference.stagedOutput) { ready = true; break }
                try await Task.sleep(for: .milliseconds(25))
            }
            try check(ready, "Fake worker never wrote partial output")
            let reopened = NativeJobStore(directory: live.directory)
            restored = reopened
            try check(reopened.busy && reopened.workerCleanupPending && reopened.jobs.first?.state == "cleanup_pending", "Live worker was marked interrupted on restore")
            await reopened.refreshWorkerCleanup()
            try check(reopened.busy, "Live worker released admission")
            do { _ = try await reopened.generate(modelURL: root, request: original); throw NativeFailure(message: "Recovery admitted a new job") }
            catch { try check(reopened.jobs.count == 1, "Blocked recovery created another job") }
        } catch { recoveryFailure = error }
        live.cancel()
        do { _ = try await task.value; throw NativeFailure(message: "Cancelled fake worker succeeded") }
        catch is CancellationError {}
        try check(live.jobs[0].state == "cancelled" && !live.busy, "Cancellation did not settle")
        if let recoveryFailure { throw recoveryFailure }
        guard let restored else { throw NativeFailure(message: "Missing recovered store") }
        await restored.refreshWorkerCleanup()
        try check(!restored.busy && !restored.workerCleanupPending && restored.jobs[0].state == "interrupted", "Confirmed old worker did not unblock recovery")
        try check(!fm.fileExists(atPath: original.output), "Cancelled worker published output")
        for started in [false, true] {
            let folder = root.appendingPathComponent(started ? "unknown-launch" : "never-started")
            try fm.createDirectory(at: folder.appendingPathComponent("outputs"), withIntermediateDirectories: true)
            var pendingRequest = NativeRequest(prompt: "pending", output: folder.appendingPathComponent("outputs/final.png").path)
            pendingRequest.model = "z-image-turbo"
            let staging = folder.appendingPathComponent("outputs/.tc-image-staging-\(UUID())")
            try fm.createDirectory(at: staging, withIntermediateDirectories: false)
            let reference = PublicImageWorker.Reference(requestID: UUID(), requestDigest: String(repeating: "a", count: 64),
                runtimeFingerprint: NativeEngine.runtimeBuildIdentity(), stagedOutput: staging.appendingPathComponent("output.png").path,
                started: started, exitConfirmed: false)
            let job = NativeJob(id: UUID(), createdAt: Date(), request: pendingRequest, state: "preparing", phase: "prepare",
                completed: 0, total: 1, elapsed: 0, publicWorker: reference)
            try JSONEncoder().encode([job]).write(to: folder.appendingPathComponent("jobs.json"))
            let recovered = NativeJobStore(directory: folder)
            try check(recovered.busy == started && recovered.workerCleanupPending == started, "Missing-journal launch state misclassified")
            try check(recovered.jobs[0].state == (started ? "cleanup_pending" : "interrupted"), "Unconfirmed launch reported terminal")
            await recovered.refreshWorkerCleanup()
            try check(recovered.busy == started, "Unknown journal unblocked on refresh")
            try check(fm.fileExists(atPath: staging.path) == started, "Unsafe or missing staging cleanup")
        }
        let corrupt = root.appendingPathComponent("corrupt-history")
        try fm.createDirectory(at: corrupt, withIntermediateDirectories: true)
        let bytes = Data("broken history".utf8)
        try bytes.write(to: corrupt.appendingPathComponent("jobs.json"))
        let unreadable = NativeJobStore(directory: corrupt)
        try check(unreadable.busy && unreadable.workerCleanupPending && unreadable.storageError != nil, "Unreadable history admitted GPU work")
        await unreadable.refreshWorkerCleanup()
        let preserved = try Data(contentsOf: corrupt.appendingPathComponent("jobs.json"))
        try check(unreadable.busy && preserved == bytes, "Recovery overwrote unreadable history")
        print("PASS: public image worker success/mismatched ID/hash/PNG rejection, cancel, live/unknown/unstarted worker recovery and unreadable-history admission (fake worker)")
    }
}
