import Foundation

@main struct PublicImageQueryTests {
    static func main() async throws {
        var legacy = NativeRequest(prompt: "query fixture", output: "/tmp/query-only/output.png")
        legacy.model = "z-image-turbo"; legacy.steps = 9
        let request = NativeRequestV2(legacy: legacy, targetBytes: 10 << 30)
        if CommandLine.arguments.count == 4 && CommandLine.arguments[1] == "--real" {
            let executable = URL(fileURLWithPath: CommandLine.arguments[2])
            let root = URL(fileURLWithPath: CommandLine.arguments[3])
            guard !FileManager.default.fileExists(atPath: root.path) else { throw NativeFailure(message: "Experiment already exists") }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            for (model, path) in [("z-image-turbo", "/Users/chencanhui/models/TurboCider/Z-Image-Turbo"), ("flux2-klein-4b", "/Users/chencanhui/models/TurboCider/FLUX.2-klein-4B")] {
                for cancel in [false, true] {
                    var intent = request; intent.model = model; intent.sampling.steps = model == "z-image-turbo" ? 9 : 4
                    let name = model + (cancel ? "-cancel" : "-reject")
                    let plan: [String: Any] = ["runtime_fingerprint": NativeEngine.runtimeBuildIdentity(), "executable": executable.path,
                        "model": model, "cancel_after_seconds": cancel ? 1 : 0, "expected": cancel ? "CancellationError after cleanup" : "catalog_has_no_public_records", "scope": "Real query adapter, no model generation"]
                    try JSONSerialization.data(withJSONObject: plan, options: .prettyPrinted).write(to: root.appendingPathComponent(name + "-plan.json"))
                    let frozen = intent
                    let task = Task { try await PublicImageQueryRunner.shared.resolve(frozen, model: URL(fileURLWithPath: path), executable: executable) }
                    if cancel { try await Task.sleep(for: .seconds(1)); task.cancel() }
                    var failure: Error?
                    do { _ = try await task.value } catch { failure = error }
                    let observation: [String: Any] = ["error": failure?.localizedDescription ?? "", "cancelled": failure is CancellationError]
                    try JSONSerialization.data(withJSONObject: observation, options: .prettyPrinted).write(to: root.appendingPathComponent(name + "-result.json"))
                    if cancel { precondition(failure is CancellationError) }
                    else { precondition(failure?.localizedDescription.contains("catalog_has_no_public_records") == true) }
                    print("Real query \(name) PASS")
                }
            }
            return
        }
        let metadata = try NativeEngine.workerStreamingOptions(request)
        let embedded = try NativeEngine.streamingOptions(request)
        precondition(metadata.execution_container == "cli_worker" && embedded.execution_container == "embedded_app")
        precondition(metadata.targets.count == 5 && metadata.targets.allSatisfy { $0.status == "catalog_empty" })
        let noOpen = try await PublicImageQueries.options(request, modelURL: URL(fileURLWithPath: "/missing/must-not-open"))
        precondition(noOpen.execution_container == "cli_worker" && noOpen.query_status == "catalog_empty")
        print("fixed-container discovery and empty-catalog no-open PASS")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tc-query-fixtures-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var completed = false
        defer {
            if completed { try? FileManager.default.removeItem(at: root) }
            else { FileHandle.standardError.write(Data("Preserved query fixtures: \(root.path)\n".utf8)) }
        }
        let marker = root.appendingPathComponent("events.jsonl")
        func executable(_ mode: String) throws -> URL {
            let url = root.appendingPathComponent("\(mode).py")
            let code = """
            #!/usr/bin/python3
            import sys,json,signal,time,os,traceback
            from pathlib import Path
            if sys.stdin.buffer.read(1)!=b'\\1':sys.exit(1)
            wire=json.loads(Path(sys.argv[2]).read_text());request=wire['native_request_v2'];mode='\(mode)'
            marker=Path('\(marker.path)');lock=marker.parent/'lock'
            def log_error(kind,value,tb):
                with (marker.parent/'failure.log').open('a') as f:traceback.print_exception(kind,value,tb,file=f)
            sys.excepthook=log_error
            lock.mkdir()
            def record(kind):
                with marker.open('a') as f:f.write(json.dumps(dict(kind=kind,mode=mode,query_root=str(Path(sys.argv[2]).parents[3])))+'\\n')
            base=dict(protocol_version=1,job_id=wire['job_id'],request_id=wire['request_id'],request_digest=wire['request_digest'],actual_container='cli_worker',runtime_fingerprint='\(NativeEngine.runtimeBuildIdentity())',artifact=None,public_streaming_summary=None,resolution_digest=None,record_digest=None,layout_digest=None,error=None)
            def stop(signum,frame):
                base.update(status='cancelled',error=dict(code='worker_cancelled',message='cancelled'))
                print(json.dumps(base),flush=True);sys.exit(2)
            signal.signal(signal.SIGTERM,stop)
            try:
                record('start')
                if mode=='wait':
                    while True:time.sleep(1)
                time.sleep(0.1)
                selector=request['execution']['streaming'];budget=selector['target_request_memory_bytes']
                exact=dict(selector,selection='preset',preset_id='fixture',preset_revision=1,catalog_revision='catalog',expected_resolution_digest='resolution')
                selection=dict(preset_id='fixture',preset_revision=1,record_digest='record',release_channel='public',target_request_memory_bytes=budget,calibrated_request_bytes=1024,memory_scope='request',layout_digest='layout',component_policy_revision='policy',execution_container='embedded_app' if mode=='wrong_container' else 'cli_worker')
                requested=dict(selector)
                if mode=='wrong_selector':requested['target_request_memory_bytes']+=1
                resolution=dict(schema_version=1,status='resolved',request_digest='workload',resolution_digest='resolution',catalog_revision='catalog',requested_selector=requested,exact_selector=exact,selection=selection,identity=dict(source_digest='source',runtime_digest='runtime',device_digest='device'))
                base.update(status='resolved',resolution=resolution,resolution_digest='resolution',record_digest='record',layout_digest='layout')
                if mode=='wrong_runtime':base['runtime_fingerprint']='other'
                print(json.dumps(base),flush=True)
            finally:
                record('end');lock.rmdir()
            """
            try Data(code.utf8).write(to: url)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
            return url
        }
        let runner = PublicImageQueryRunner()
        let normal = try executable("normal")
        async let first = runner.resolve(request, model: root, executable: normal)
        async let second = runner.resolve(request, model: root, executable: normal)
        let resolved = try await (first, second)
        precondition(resolved.0.selection.execution_container == "cli_worker" && resolved.1.status == "resolved")
        for mode in ["wrong_container", "wrong_runtime", "wrong_selector"] {
            var failure: Error?
            do { _ = try await runner.resolve(request, model: root, executable: executable(mode)) } catch { failure = error }
            precondition(failure != nil)
        }
        let wait = try executable("wait")
        let task = Task { try await runner.resolve(request, model: root, executable: wait) }
        var started = false
        for _ in 0..<200 {
            if let data = try? String(contentsOf: marker, encoding: .utf8), data.contains("\"mode\": \"wait\"") { started = true; break }
            try await Task.sleep(for: .milliseconds(25))
        }
        let queued = Task { try await runner.resolve(request, model: root, executable: normal) }
        try await Task.sleep(for: .milliseconds(50)); queued.cancel()
        var queueCancelled = false
        do { _ = try await queued.value } catch is CancellationError { queueCancelled = true }
        task.cancel()
        var activeCancelled = false
        do { _ = try await task.value } catch is CancellationError { activeCancelled = true }
        precondition(started && queueCancelled && activeCancelled)
        _ = try await runner.resolve(request, model: root, executable: normal)
        let lines = try String(contentsOf: marker, encoding: .utf8).split(separator: "\n")
        var active = 0, starts = 0
        for line in lines {
            let event = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            if event["kind"] as? String == "start" { active += 1; starts += 1 } else { active -= 1 }
            precondition((0...1).contains(active))
            precondition(!FileManager.default.fileExists(atPath: event["query_root"] as! String))
        }
        precondition(active == 0 && starts == 7)
        completed = true
        print("query correlation/container/runtime/selector, serialization, active/queued cancellation and temporary cleanup PASS (fake workers)")
    }
}
