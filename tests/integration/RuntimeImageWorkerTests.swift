import Foundation
import Darwin

@main struct RuntimeImageWorkerTests {
    static func main() throws {
        // Cross-language fixtures from the native worker contract test.
        if CommandLine.arguments.count == 3 {
            let inputURL = URL(fileURLWithPath: CommandLine.arguments[1])
            let input = try Data(contentsOf: inputURL)
            let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
            let request = try JSONDecoder().decode(NativeRequestV2.self, from: JSONSerialization.data(withJSONObject: wire["native_request_v2"]!))
            let options = try JSONDecoder().decode(RuntimeImageWorker.Options.self, from: JSONSerialization.data(withJSONObject: wire["runtime_options"]!))
            let reference = PublicImageWorker.Reference(requestID: UUID(uuidString: wire["request_id"] as! String)!,
                requestDigest: wire["request_digest"] as! String, runtimeFingerprint: "test-runtime", stagedOutput: request.outputs[0].path)
            let prepared = RuntimeImageWorker.Prepared(reference: reference, input: input, inputURL: inputURL, request: request, options: options)
            let result = try RuntimeImageWorker.validateTerminal(Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])), prepared: prepared, exitCode: 0)
            precondition(result.status == "succeeded" && result.receipt != nil && result.artifact != nil)
            print("Swift Runtime terminal accepted native fixture PASS")
            return
        }
        var checks = 0
        func check(_ value: Bool, _ reason: String) throws {
            guard value else { throw NativeFailure(message: reason) }; checks += 1
        }
        func rejects(_ action: () throws -> Void) throws {
            do { try action() } catch { checks += 1; return }
            throw NativeFailure(message: "Invalid Runtime configuration accepted")
        }
        let root = URL(fileURLWithPath: "/private/tmp/tc-runtime-worker-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let options = try RuntimeImageWorker.options(model: "z-image-turbo", store: root)
        let bytes = try Data(contentsOf: URL(fileURLWithPath: options.descriptor_path))
        try check(bytes.count < 1024, "Descriptor exported weights instead of a small shape")
        try check(try RuntimeImageWorker.options(model: "z-image-turbo", store: root) == options, "Descriptor reuse changed identity")
        try check(try Data(contentsOf: URL(fileURLWithPath: options.descriptor_path)) == bytes, "Descriptor reuse rewrote bytes")
        var legacy = NativeRequest(prompt: "CPU fixture", output: root.appendingPathComponent("output.png").path)
        legacy.model = "z-image-turbo"; legacy.operation = "image.generate"; legacy.execution = "gpu_ane"
        legacy.steps = 8; legacy.residency = "resident"; legacy.ane_manifest = options.descriptor_path
        legacy.allow_approximation = true; legacy.hybrid_mlp_mode = "runtime"; legacy.lora_strategy = "auto"
        let jobID = UUID()
        let prepared = try RuntimeImageWorker.prepare(jobID: jobID, model: root, legacy: legacy, options: options, store: root)
        try check(prepared.request.execution.residency == "resident" && prepared.request.lora_strategy == nil,
                  "Runtime conversion lost residency or leaked base LoRA strategy")
        let wire = try JSONSerialization.jsonObject(with: prepared.input) as! [String: Any]
        let native = wire["native_request_v2"] as! [String: Any]
        let typed = wire["runtime_options"] as! [String: Any]
        try check(try RuntimeImageWorker.requestDigest(request: native, options: typed) == prepared.reference.requestDigest, "Digest differs from submitted intent")
        var changed = typed; changed["cache_dir"] = "/private/tmp/other"
        try check(try RuntimeImageWorker.requestDigest(request: native, options: changed) != prepared.reference.requestDigest, "Options omitted from digest")
        var changedNative = native; changedNative["sampling"] = ["seed": 43, "steps": 8]
        try check(try RuntimeImageWorker.requestDigest(request: changedNative, options: typed) != prepared.reference.requestDigest, "Native intent omitted from digest")
        let environment = try RuntimeImageWorker.environment(options: options, model: "z-image-turbo", inherited: [
            "PATH": "/usr/bin", "TURBOCIDER_ANE_BACKEND": "gpu", "TURBOCIDER_CAPTURE": "1", "TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC": "1"])
        try check(environment["PATH"] == "/usr/bin" && environment["TURBOCIDER_ANE_BACKEND"] == "auto" &&
                  environment["TURBOCIDER_CAPTURE"] == nil && environment["TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC"] == nil, "Ambient flags escaped child allowlist")
        let qwenOptions = try RuntimeImageWorker.options(model: "qwen-image-2.1", store: root)
        let qwenEnvironment = try RuntimeImageWorker.environment(options: qwenOptions, model: "qwen-image-2.1", inherited: [:])
        try check(qwenEnvironment["TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC"] == "1" &&
                  qwenEnvironment["TURBOCIDER_QWEN21_RUNTIME_PREPARE_EARLY"] == "0", "Qwen staged child profile missing")
        try check(RuntimeImageWorker.hardwareEligible(systemJSON: #"{"gpu":"Apple M4 Pro","physical_memory_bytes":51539607552}"#) &&
                  !RuntimeImageWorker.hardwareEligible(systemJSON: #"{"gpu":"Apple M4 Pro","physical_memory_bytes":25769803776}"#), "Hardware experiment admission broadened")
        var invalid = prepared.request
        invalid.sampling.steps = 9
        try rejects { try RuntimeImageWorker.validate(request: invalid, options: options) }
        invalid = prepared.request; invalid.execution.streaming = NativeStreamingSelectorV2(targetBytes: 10 << 30)
        try rejects { try RuntimeImageWorker.validate(request: invalid, options: options) }
        invalid = prepared.request; invalid.parameters.dynamic_text = false
        try rejects { try RuntimeImageWorker.validate(request: invalid, options: options) }
        var qwen = prepared.request
        qwen.model = "qwen-image-2.1"; qwen.sampling.steps = 20; qwen.execution.residency = "component_staged"
        qwen.execution.ane_manifest = qwenOptions.descriptor_path; qwen.parameters.qwen21_reference_size = 1024
        try RuntimeImageWorker.validate(request: qwen, options: qwenOptions); checks += 1
        for count in 1...2 {
            qwen.operation = "image.edit"
            qwen.inputs = [qwen.inputs[0]] + (0..<count).map { NativeInputV2(kind: "image", role: "reference", path: root.appendingPathComponent("ref\($0).png").path, text: nil, strength: nil) }
            try RuntimeImageWorker.validate(request: qwen, options: qwenOptions); checks += 1
        }
        qwen.inputs.append(qwen.inputs[1])
        try rejects { try RuntimeImageWorker.validate(request: qwen, options: qwenOptions) }
        qwen.inputs = [qwen.inputs[0]]; qwen.operation = "image.generate"; qwen.sampling.steps = 6
        qwen.lora_strategy = "inference_time"
        qwen.loras = [NativeLoRA(path: root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors").path)]
        try RuntimeImageWorker.validate(request: qwen, options: qwenOptions); checks += 1
        qwen.loras![0].strength = 0.9
        try rejects { try RuntimeImageWorker.validate(request: qwen, options: qwenOptions) }
        qwen.loras![0].strength = 1
        qwen.loras![0].path = root.appendingPathComponent("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors").path
        try rejects { try RuntimeImageWorker.validate(request: qwen, options: qwenOptions) }

        func terminal(hybridBlocks: Int = 32, fallbackBlocks: Int = 0, selected: Bool = true) -> [String: Any] {
            let reason = fallbackBlocks > 0 || !selected || hybridBlocks == 0 ? "fixture GPU fallback" : ""
            let selectedBackend: Any = selected ? "private_ane" : NSNull()
            let dataPath = selected ? "w8a8_hadamard" : "", axis = selected ? "intermediate_channels" : "rows"
            let receipt: [String: Any] = ["backend": hybridBlocks > 0 ? "private_ane" : "gpu", "selected_backend": selectedBackend,
                "actual_execution": hybridBlocks > 0 ? "gpu_ane" : "gpu", "partial_fallback": hybridBlocks > 0 && fallbackBlocks > 0,
                "data_path": hybridBlocks > 0 ? dataPath : "", "partition_axis": hybridBlocks > 0 ? axis : "", "ane_channels": hybridBlocks > 0 ? 4096 : 0,
                "fallback_reason": reason, "hybrid_blocks": hybridBlocks, "gpu_blocks": 1, "fallback_blocks": fallbackBlocks]
            let metrics: [String: Any] = ["executor_backend": selectedBackend, "data_path": dataPath, "partition_axis": axis,
                "ane_channels": selected ? 4096 : 0, "gpu_channels": selected ? 6144 : 0, "backend_fallback_reason": reason, "failure_reason": "",
                "hybrid_blocks_session_total": hybridBlocks, "gpu_blocks_session_total": 1, "fallback_blocks_session_total": fallbackBlocks]
            let result: [String: Any] = ["schema_version": 1, "warmup": false, "model": legacy.model, "operation": "image.generate", "output": legacy.output,
                "width": 512, "height": 512, "steps": 8, "seed": 42,
                "plan": ["execution": selected ? "gpu_ane_experimental" : "gpu", "runtime_weight_contract": ["executor_backend": selectedBackend, "data_path": dataPath, "partition_axis": axis]],
                "hybrid": ["runtime_weight": metrics, "runtime_failed": false]]
            return ["protocol_version": 1, "job_id": wire["job_id"]!, "request_id": wire["request_id"]!, "request_digest": wire["request_digest"]!,
                "actual_container": "cli_worker", "runtime_fingerprint": prepared.reference.runtimeFingerprint, "status": "succeeded", "error": NSNull(),
                "runtime_options": typed, "runtime_receipt": receipt, "result": result,
                "artifact": ["path": legacy.output, "size": 123, "sha256": String(repeating: "a", count: 64)],
                "resolution_digest": NSNull(), "record_digest": NSNull(), "layout_digest": NSNull(), "public_streaming_summary": NSNull()]
        }
        func verify(_ terminal: [String: Any], code: Int32 = 0) throws -> RuntimeImageWorker.Verified {
            try RuntimeImageWorker.validateTerminal(JSONSerialization.data(withJSONObject: terminal), prepared: prepared, exitCode: code)
        }
        let success = terminal()
        try check(try verify(success).receipt?.actual_execution == "gpu_ane", "Private execution receipt lost")
        try check(try verify(terminal(hybridBlocks: 0, fallbackBlocks: 32, selected: false)).receipt?.label.contains("GPU · Runtime 实验回退") == true, "Unselected executor fallback mislabelled")
        try check(try verify(terminal(hybridBlocks: 0, fallbackBlocks: 32)).receipt?.backend == "gpu", "All-fallback selected executor was presented as actual ANE")
        try check(try verify(terminal(fallbackBlocks: 2)).receipt?.partial_fallback == true, "Partial fallback was hidden")
        for key in ["job_id", "request_id", "request_digest", "runtime_fingerprint", "actual_container"] {
            var wrong = success; wrong[key] = "other"
            try rejects { _ = try verify(wrong) }
        }
        var wrong = success; wrong["runtime_options"] = changed
        try rejects { _ = try verify(wrong) }
        wrong = success; wrong["public_streaming_summary"] = [:]
        try rejects { _ = try verify(wrong) }
        wrong = success
        var receipt = wrong["runtime_receipt"] as! [String: Any]; receipt["hybrid_blocks"] = true; wrong["runtime_receipt"] = receipt
        try rejects { _ = try verify(wrong) }
        wrong = success; receipt = wrong["runtime_receipt"] as! [String: Any]; receipt["data_path"] = "fp16"; wrong["runtime_receipt"] = receipt
        try rejects { _ = try verify(wrong) }
        wrong = success; receipt = wrong["runtime_receipt"] as! [String: Any]; receipt["actual_execution"] = "gpu"; wrong["runtime_receipt"] = receipt
        try rejects { _ = try verify(wrong) }
        wrong = success; receipt = wrong["runtime_receipt"] as! [String: Any]; receipt["partial_fallback"] = 0; wrong["runtime_receipt"] = receipt
        try rejects { _ = try verify(wrong) }
        wrong = success
        var result = wrong["result"] as! [String: Any]
        var hybrid = result["hybrid"] as! [String: Any]
        var metrics = hybrid["runtime_weight"] as! [String: Any]
        metrics["gpu_blocks_session_total"] = true; hybrid["runtime_weight"] = metrics; result["hybrid"] = hybrid; wrong["result"] = result
        try rejects { _ = try verify(wrong) }
        wrong = success; result = wrong["result"] as! [String: Any]; result["warmup"] = 0; wrong["result"] = result
        try rejects { _ = try verify(wrong) }
        // Exporter keeps failure_reason inside runtime_weight. A fake top-level
        // field must neither substitute for missing provenance nor override it.
        wrong = success; result = wrong["result"] as! [String: Any]; hybrid = result["hybrid"] as! [String: Any]
        metrics = hybrid["runtime_weight"] as! [String: Any]; metrics["failure_reason"] = nil
        hybrid["runtime_weight"] = metrics; hybrid["failure_reason"] = "fake top-level"
        result["hybrid"] = hybrid; wrong["result"] = result
        try rejects { _ = try verify(wrong) }
        wrong = success; result = wrong["result"] as! [String: Any]; hybrid = result["hybrid"] as! [String: Any]
        hybrid["failure_reason"] = "fake top-level"; result["hybrid"] = hybrid; wrong["result"] = result
        try check(try verify(wrong).receipt == verify(success).receipt, "Fake top-level reason overrode nested native provenance")
        wrong = success; result = wrong["result"] as! [String: Any]; hybrid = result["hybrid"] as! [String: Any]
        metrics = hybrid["runtime_weight"] as! [String: Any]; metrics["failure_reason"] = "native runtime failed"
        hybrid["runtime_weight"] = metrics; hybrid["runtime_failed"] = true; result["hybrid"] = hybrid; wrong["result"] = result
        receipt = wrong["runtime_receipt"] as! [String: Any]; receipt["fallback_reason"] = "native runtime failed"; receipt["partial_fallback"] = true
        wrong["runtime_receipt"] = receipt
        try check(try verify(wrong).receipt?.fallback_reason == "native runtime failed", "Nested Runtime failure reason was not used")
        let storedOptions = try JSONDecoder().decode(RuntimeImageWorker.Options.self, from: JSONEncoder().encode(prepared.options))
        let storedReference = try JSONDecoder().decode(PublicImageWorker.Reference.self, from: JSONEncoder().encode(prepared.reference))
        let recovered = RuntimeImageWorker.Prepared(reference: storedReference, input: prepared.input, inputURL: prepared.inputURL,
                                                   request: prepared.request, options: storedOptions)
        let recoveredTerminal = try RuntimeImageWorker.validateTerminal(JSONSerialization.data(withJSONObject: success), prepared: recovered, exitCode: 0)
        try check(recoveredTerminal.receipt == verify(success).receipt, "Persisted Options/Reference cannot reconstruct terminal correlation")
        var error = success
        error["status"] = "error"; error["result"] = nil; error["runtime_receipt"] = NSNull(); error["artifact"] = NSNull()
        error["error"] = ["code": "runtime_unavailable", "message": "fixture"]
        try check(try verify(error, code: 1).status == "error", "Valid worker error rejected")
        error["status"] = "cancelled"; error["error"] = ["code": "worker_cancelled", "message": "fixture"]
        try check(try verify(error, code: 2).status == "cancelled", "Valid cancellation rejected")
        let events = RuntimeWorkerEventStream(jobID: jobID, reference: prepared.reference) { _ in }
        let event: [String: Any] = ["event_schema_version": 1, "kind": "progress", "sequence": 1, "job_id": wire["job_id"]!, "request_id": wire["request_id"]!,
            "request_digest": wire["request_digest"]!, "execution_container": "cli_worker", "runtime_fingerprint": prepared.reference.runtimeFingerprint,
            "payload": ["sequence": 0, "phase": "prepare", "completed": 0, "total": 1, "elapsed_seconds": 0]]
        var line = Data("TC_EVENT\t".utf8); line.append(try JSONSerialization.data(withJSONObject: event)); line.append(10)
        try events.consume(Data(line.prefix(17))); try events.consume(Data(line.dropFirst(17))); try events.finish(); checks += 1
        try rejects { try events.consume(line) }
        let partial = RuntimeWorkerEventStream(jobID: jobID, reference: prepared.reference) { _ in }
        try partial.consume(Data("TC_EVENT\t{".utf8)); try rejects { try partial.finish() }
        try Data("{}".utf8).write(to: URL(fileURLWithPath: options.descriptor_path))
        try rejects { _ = try RuntimeImageWorker.options(model: "z-image-turbo", store: root) }
        checks += try verifyProgramCache(root: root)
        print("PASS Runtime App contracts: \(checks) CPU checks; no model loaded, no GPU/ANE execution")
    }
    private static func verifyProgramCache(root: URL) throws -> Int {
        let fm = FileManager.default
        let store = root.appendingPathComponent("bounded-program-cache")
        try fm.createDirectory(at: store, withIntermediateDirectories: true)
        let options = try RuntimeImageWorker.options(model: "z-image-turbo", store: store)
        let cache = URL(fileURLWithPath: options.cache_dir)
        var checks = 0
        func check(_ value: Bool, _ reason: String) throws {
            guard value else { throw NativeFailure(message: reason) }; checks += 1
        }
        func rejects(_ action: () throws -> Void) throws {
            do { try action() } catch { checks += 1; return }
            throw NativeFailure(message: "Unsafe program cache cleanup accepted")
        }
        let a = String(repeating: "a", count: 64), b = String(repeating: "b", count: 64)
        let c = String(repeating: "c", count: 64), d = String(repeating: "d", count: 64)
        func entry(_ key: String, bytes: Int = 20, read: Int, modified: Int = 1) throws {
            let directory = cache.appendingPathComponent(key)
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(repeating: 109, count: 5).write(to: directory.appendingPathComponent("model.mil"))
            try Data(repeating: 1, count: bytes - 5).write(to: directory.appendingPathComponent("weights.bin"))
            try Data().write(to: cache.appendingPathComponent(key + ".lock"))
            for name in ["model.mil", "weights.bin"] {
                let stamps = [timeval(tv_sec: read, tv_usec: 0), timeval(tv_sec: modified, tv_usec: 0)]
                let result = stamps.withUnsafeBufferPointer { utimes(directory.appendingPathComponent(name).path, $0.baseAddress) }
                guard result == 0 else { throw NativeFailure(message: "Could not set CPU fixture access time") }
            }
        }
        func trim(_ bytes: UInt64, _ count: Int = 64, idle: Bool = true) throws -> RuntimeImageWorker.ProgramCacheReport {
            try RuntimeImageWorker.trimProgramCache(store: store, workerIsIdle: idle,
                budget: .init(maximumBytes: bytes, maximumEntries: count))
        }
        try check(RuntimeImageWorker.ProgramCacheBudget.standard.maximumBytes == 128 << 20 &&
                  RuntimeImageWorker.ProgramCacheBudget.standard.maximumEntries == 64, "Production cache budget changed")
        try entry(a, read: 10); try entry(b, read: 20); try entry(c, read: 30)
        try rejects { _ = try trim(0, 0, idle: false) }
        try check(fm.fileExists(atPath: cache.appendingPathComponent(a).path), "Busy cleanup deleted a live entry")
        let byBytes = try trim(40)
        try check(byBytes.beforeBytes == 60 && byBytes.afterBytes == 40 && byBytes.afterEntries == 2 && byBytes.evictedKeys == [a],
                  "Byte budget did not evict least-recently-read source")
        try check(!fm.fileExists(atPath: cache.appendingPathComponent(a + ".lock").path) &&
                  fm.fileExists(atPath: options.descriptor_path), "Eviction leaked a lock or deleted descriptor")
        let byCount = try trim(UInt64.max, 1)
        try check(byCount.evictedKeys == [b] && byCount.afterBytes == 20 && byCount.afterEntries == 1, "Entry cap did not retain newest source")
        try entry(d, bytes: 45, read: 40)
        let oversized = try trim(35)
        try check(oversized.evictedKeys == [c, d] && oversized.afterBytes == 0, "Oversized single entry exceeded cache cap")
        try entry(b, read: 1); try entry(a, read: 1)
        let tie = try trim(20)
        try check(tie.evictedKeys == [a], "LRU tie ordering is not deterministic")
        try Data().write(to: cache.appendingPathComponent(c + ".lock"))
        _ = try trim(20)
        try check(!fm.fileExists(atPath: cache.appendingPathComponent(c + ".lock").path), "Unused native lock accumulated outside entry cap")
        let note = cache.appendingPathComponent("notes.txt")
        try Data("keep".utf8).write(to: note)
        try rejects { _ = try trim(0, 0) }
        try check(fm.fileExists(atPath: cache.appendingPathComponent(b).path) && (try Data(contentsOf: note)) == Data("keep".utf8),
                  "Unknown root contents were deleted before validation completed")
        try fm.removeItem(at: note)
        let extra = cache.appendingPathComponent(b + "/extra.log")
        try Data([1]).write(to: extra)
        try rejects { _ = try trim(0, 0) }
        try check(fm.fileExists(atPath: extra.path), "Unknown nested contents were recursively removed")
        try fm.removeItem(at: extra)
        let external = root.appendingPathComponent("external-keep.bin")
        try Data("external".utf8).write(to: external)
        let weights = cache.appendingPathComponent(b + "/weights.bin")
        try fm.removeItem(at: weights)
        try fm.createSymbolicLink(atPath: weights.path, withDestinationPath: external.path)
        try rejects { _ = try trim(0, 0) }
        try check(try Data(contentsOf: external) == Data("external".utf8), "Symlink target outside cache was changed")
        try fm.removeItem(at: weights)
        guard link(external.path, weights.path) == 0 else { throw NativeFailure(message: "Hard-link CPU fixture failed") }
        try rejects { _ = try trim(0, 0) }
        try check(fm.fileExists(atPath: cache.appendingPathComponent(b + "/model.mil").path), "Hard-link entry was partially removed")
        try fm.removeItem(at: weights); try Data(repeating: 1, count: 15).write(to: weights)
        let foreignDirectory = root.appendingPathComponent("foreign-directory")
        try fm.createDirectory(at: foreignDirectory, withIntermediateDirectories: true)
        try fm.createSymbolicLink(atPath: cache.appendingPathComponent(d).path, withDestinationPath: foreignDirectory.path)
        try Data().write(to: cache.appendingPathComponent(d + ".lock"))
        try rejects { _ = try trim(0, 0) }
        try check(fm.fileExists(atPath: foreignDirectory.path), "Linked directory outside cache was removed")
        try fm.removeItem(at: cache.appendingPathComponent(d)); try fm.removeItem(at: cache.appendingPathComponent(d + ".lock"))
        let lockFD = open(cache.appendingPathComponent(b + ".lock").path, O_RDWR | O_NOFOLLOW)
        guard lockFD >= 0, flock(lockFD, LOCK_EX | LOCK_NB) == 0 else { throw NativeFailure(message: "Cache lock CPU fixture failed") }
        do {
            defer { flock(lockFD, LOCK_UN); close(lockFD) }
            try rejects { _ = try trim(0, 0) }
            try check(fm.fileExists(atPath: weights.path), "Locked program source was deleted")
        }
        let cleared = try trim(0, 0)
        try check(cleared.afterEntries == 0 && cleared.afterBytes == 0 && fm.fileExists(atPath: options.descriptor_path),
                  "Cache clear did not preserve descriptor")
        func partial(_ key: String, modelBytes: Int?, weightBytes: Int?) throws {
            let directory = cache.appendingPathComponent(key)
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            if let modelBytes { try Data(repeating: 109, count: modelBytes).write(to: directory.appendingPathComponent("model.mil")) }
            if let weightBytes { try Data(repeating: 1, count: weightBytes).write(to: directory.appendingPathComponent("weights.bin")) }
            try Data().write(to: cache.appendingPathComponent(key + ".lock"))
        }
        let e = String(repeating: "e", count: 64)
        try partial(a, modelBytes: nil, weightBytes: nil)
        try partial(b, modelBytes: 5, weightBytes: nil)
        try partial(c, modelBytes: nil, weightBytes: 3)
        try partial(d, modelBytes: 0, weightBytes: 7)
        try partial(e, modelBytes: 0, weightBytes: 0)
        let recovered = try trim(UInt64.max)
        try check(recovered.beforeBytes == 15 && recovered.beforeEntries == 5 && recovered.afterEntries == 0 &&
                  recovered.afterBytes == 0 && Set(recovered.evictedKeys) == Set([a, b, c, d, e]),
                  "Known interrupted source writes did not recover below the byte budget")
        try check(fm.fileExists(atPath: options.descriptor_path) && (try fm.contentsOfDirectory(atPath: cache.path)).isEmpty,
                  "Partial recovery leaked source locks or removed the descriptor")
        try entry(a, bytes: 5, read: 1)
        let noConstants = try trim(UInt64.max)
        try check(noConstants.afterEntries == 1 && noConstants.afterBytes == 5 && noConstants.evictedKeys.isEmpty,
                  "A valid no-constants program was mistaken for an interrupted source")
        try partial(b, modelBytes: 5, weightBytes: nil)
        let partialLock = open(cache.appendingPathComponent(b + ".lock").path, O_RDWR | O_NOFOLLOW)
        guard partialLock >= 0, flock(partialLock, LOCK_EX | LOCK_NB) == 0 else { throw NativeFailure(message: "Partial cache lock CPU fixture failed") }
        do {
            defer { flock(partialLock, LOCK_UN); close(partialLock) }
            try rejects { _ = try trim(UInt64.max) }
            try check(fm.fileExists(atPath: cache.appendingPathComponent(b + "/model.mil").path),
                      "Locked interrupted source was removed")
        }
        let unlocked = try trim(UInt64.max)
        try check(unlocked.evictedKeys == [b] && unlocked.afterEntries == 1, "Unlocked interrupted source could not recover")
        try partial(d, modelBytes: nil, weightBytes: nil)
        let unknownPartial = cache.appendingPathComponent(d + "/unknown.bin")
        try Data([7]).write(to: unknownPartial)
        try rejects { _ = try trim(0, 0) }
        try check(fm.fileExists(atPath: unknownPartial.path) && fm.fileExists(atPath: cache.appendingPathComponent(a).path),
                  "Unknown partial contents bypassed full validation")
        try fm.removeItem(at: unknownPartial)
        let partialWeights = cache.appendingPathComponent(d + "/weights.bin")
        try fm.createSymbolicLink(atPath: partialWeights.path, withDestinationPath: external.path)
        try rejects { _ = try trim(UInt64.max) }
        try check(try Data(contentsOf: external) == Data("external".utf8), "Partial symlink escaped the owned directory")
        try fm.removeItem(at: partialWeights)
        guard link(external.path, partialWeights.path) == 0 else { throw NativeFailure(message: "Partial hard-link CPU fixture failed") }
        try rejects { _ = try trim(UInt64.max) }
        try check(fm.fileExists(atPath: partialWeights.path), "Partial hard link was removed")
        try fm.removeItem(at: partialWeights)
        _ = try trim(0, 0)
        let aliasStore = root.appendingPathComponent("aliased-program-cache")
        let alias = aliasStore.appendingPathComponent("cache/runtime-ane/programs")
        try fm.createDirectory(at: alias.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fm.createSymbolicLink(atPath: alias.path, withDestinationPath: cache.path)
        try rejects { _ = try RuntimeImageWorker.trimProgramCache(store: aliasStore, workerIsIdle: true, budget: .init(maximumBytes: 0, maximumEntries: 0)) }
        try check(fm.fileExists(atPath: cache.path), "Symlink cache root escaped into another directory")
        print("PASS bounded Runtime source cache: \(checks) CPU checks, tiny fixtures, 128 MiB / 64-entry default, LRU/partial recovery/locks/link rejection/descriptor retention")
        return checks
    }
}
