import Foundation

@main struct WorkerTerminalEnvelopeTests {
    static func main() throws {
        if CommandLine.arguments.count == 3 {
            let input = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
            let terminal = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
            let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
            let request = try JSONDecoder().decode(NativeRequestV2.self,
                from: JSONSerialization.data(withJSONObject: wire["native_request_v2"]!))
            let verified = try WorkerTerminalEnvelope.validate(terminal, input: input, request: request,
                runtimeFingerprint: "test-runtime", operation: .generate, exitCode: 0)
            precondition(verified.status == "succeeded" && verified.artifact != nil && verified.result != nil)
            print("Native wrapper terminal accepted by Swift PASS (C API doubles)")
            return
        }
        let request = NativeRequestV2(legacy: NativeRequest(prompt: "fixture", output: "/tmp/job/output.png"), targetBytes: 10 << 30)
        let input = try WorkerRequestEnvelope.encode(jobID: UUID(), requestID: UUID(),
            installation: URL(fileURLWithPath: "/tmp/model"), nativeRequest: JSONEncoder().encode(request))
        let wire = try JSONSerialization.jsonObject(with: input) as! [String: Any]
        var exact = request.execution.streaming!
        exact.selection = "preset"; exact.preset_id = "fixture"; exact.preset_revision = 3
        exact.catalog_revision = "catalog"; exact.expected_resolution_digest = "resolution"
        let resolution = NativeStreamingResolution(schema_version: 1, status: "resolved", request_digest: "workload",
            resolution_digest: "resolution", catalog_revision: "catalog", requested_selector: request.execution.streaming!, exact_selector: exact,
            selection: NativeStreamingResolutionSelection(preset_id: "fixture", preset_revision: 3, record_digest: "record",
                release_channel: "public", target_request_memory_bytes: 10 << 30, calibrated_request_bytes: 8 << 30,
                memory_scope: "request", layout_digest: "layout", component_policy_revision: "policy", execution_container: "cli_worker"),
            identity: NativeStreamingResolutionIdentity(source_digest: "source", runtime_digest: "runtime", device_digest: "device"))
        let summary: [String: Any] = ["schema_version": 1, "target_request_memory_bytes": 10 << 30,
            "calibrated_request_bytes": 8 << 30, "preset_id": "fixture", "preset_revision": 3,
            "catalog_revision": "catalog", "record_digest": "record", "resolution_digest": "resolution",
            "source_digest": "source", "workload_digest": "workload", "runtime_digest": "runtime", "device_digest": "device",
            "authorized_layout_digest": "layout", "actual_layout_digest": "layout", "component_policy_revision": "policy",
            "execution_container": "cli_worker", "memory_scope": "request", "receipt_schema_version": 3,
            "receipt_source_generation": 1, "receipt_digest": "receipt", "receipt_verifier_revision": "verifier", "actual_plan_verified": true]
        let result: [String: Any] = ["schema_version": 1, "model": request.model, "operation": request.operation,
            "output": request.outputs[0].path, "width": request.outputs[0].width, "height": request.outputs[0].height,
            "steps": request.sampling.steps, "seed": request.sampling.seed, "warmup": false, "public_streaming": summary]
        let artifact: [String: Any] = ["path": request.outputs[0].path, "size": 123, "sha256": String(repeating: "a", count: 64)]
        let resolutionJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(resolution))
        var success: [String: Any] = ["protocol_version": 1, "job_id": wire["job_id"]!, "request_id": wire["request_id"]!,
            "request_digest": wire["request_digest"]!, "status": "succeeded", "actual_container": "cli_worker", "runtime_fingerprint": "pinned-runtime",
            "resolution_digest": "resolution", "record_digest": "record", "layout_digest": "layout", "error": NSNull(),
            "resolution": resolutionJSON, "result": result, "public_streaming_summary": summary, "artifact": artifact]
        func validate(_ value: [String: Any], operation: WorkerTerminalEnvelope.Operation = .generate, code: Int32 = 0) throws {
            _ = try WorkerTerminalEnvelope.validate(JSONSerialization.data(withJSONObject: value), input: input, request: request,
                runtimeFingerprint: "pinned-runtime", operation: operation, exitCode: code)
        }
        func rejects(_ value: [String: Any], code: Int32 = 0) throws {
            do { try validate(value, code: code) } catch { return }
            fatalError("Invalid worker terminal accepted")
        }
        try validate(success)
        for (key, value) in ["job_id": UUID().uuidString.lowercased(), "request_id": UUID().uuidString.lowercased(),
                             "request_digest": "other", "runtime_fingerprint": "other", "actual_container": "embedded_app",
                             "resolution_digest": "other", "record_digest": "other", "layout_digest": "other", "status": "resolved"] {
            var wrong = success; wrong[key] = value; try rejects(wrong)
        }
        var wrong = success; wrong["protocol_version"] = true; try rejects(wrong)
        try rejects(success, code: 1)
        try rejects([:])
        var wrongSummary = summary; wrongSummary["actual_plan_verified"] = false
        wrong = success; wrong["public_streaming_summary"] = wrongSummary; try rejects(wrong)
        var wrongResult = result; wrongResult["public_streaming"] = wrongSummary
        wrong["result"] = wrongResult; try rejects(wrong)
        for (key, value) in ["path": "/tmp/other.png", "sha256": "bad"] {
            var wrongArtifact = artifact; wrongArtifact[key] = value
            wrong = success; wrong["artifact"] = wrongArtifact; try rejects(wrong)
        }
        var query = success; query["status"] = "resolved"; query.removeValue(forKey: "result")
        query["artifact"] = NSNull(); query["public_streaming_summary"] = NSNull()
        try validate(query, operation: .query)
        var failed = query; failed["status"] = "error"; failed.removeValue(forKey: "resolution")
        for key in ["resolution_digest", "record_digest", "layout_digest"] { failed[key] = NSNull() }
        failed["error"] = ["code": "worker_generate_failed", "message": "failed"]
        try validate(failed, code: 1)
        wrong = failed; wrong["artifact"] = artifact; try rejects(wrong, code: 1)
        failed["status"] = "cancelled"; failed["error"] = ["code": "worker_cancelled", "message": "cancelled"]
        try validate(failed, code: 2)
        try rejects(failed, code: 1)
        // Helper verifies metadata only; the App must independently read/hash/decode the artifact.
        success["artifact"] = artifact
        print("Worker terminal binding/query/success/error/cancel mutation checks PASS")
    }
}
