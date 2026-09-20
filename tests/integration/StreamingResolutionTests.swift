import Foundation

@main
struct StreamingResolutionTests {
    static func main() throws {
        func require(_ value: Bool, _ message: String) throws {
            if !value { throw NativeFailure(message: message) }
        }
        func rejects(_ work: () throws -> Void) throws {
            do { try work() } catch { return }
            throw NativeFailure(message: "Expected mismatched streaming evidence to fail")
        }
        let original = NativeRequestV2(legacy: NativeRequest(prompt: "fixture", output: "/tmp/fixture.png"), targetBytes: 10 << 30)
        let requested = original.execution.streaming!
        var exact = requested
        exact.selection = "preset"; exact.preset_id = "fixture"; exact.preset_revision = 3
        exact.catalog_revision = "catalog-1"; exact.expected_resolution_digest = "resolution-1"
        let selection = NativeStreamingResolutionSelection(
            preset_id: "fixture", preset_revision: 3, record_digest: "record-1",
            release_channel: "public", target_request_memory_bytes: 10 << 30,
            calibrated_request_bytes: 8 << 30, memory_scope: "request",
            layout_digest: "layout-1", component_policy_revision: "components-1",
            execution_container: "embedded_app")
        let resolution = NativeStreamingResolution(
            schema_version: 1, status: "resolved", request_digest: "workload-1",
            resolution_digest: "resolution-1", catalog_revision: "catalog-1",
            requested_selector: requested, exact_selector: exact, selection: selection,
            identity: NativeStreamingResolutionIdentity(source_digest: "source-1", runtime_digest: "runtime-1", device_digest: "device-1"))
        let bound = try resolution.binding(original)
        try require(bound.execution.streaming == exact && original.execution.streaming == requested,
                    "Binding must freeze a copy and preserve the original intent")
        var wrongRequest = original
        wrongRequest.execution.streaming?.target_request_memory_bytes = 8 << 30
        try rejects { _ = try resolution.binding(wrongRequest) }
        let summary: [String: Any] = [
            "schema_version": 1, "target_request_memory_bytes": 10 << 30,
            "calibrated_request_bytes": 8 << 30, "preset_id": "fixture", "preset_revision": 3,
            "catalog_revision": "catalog-1", "record_digest": "record-1",
            "resolution_digest": "resolution-1", "source_digest": "source-1",
            "workload_digest": "workload-1", "runtime_digest": "runtime-1", "device_digest": "device-1",
            "authorized_layout_digest": "layout-1", "actual_layout_digest": "layout-1",
            "component_policy_revision": "components-1", "execution_container": "embedded_app",
            "memory_scope": "request", "receipt_schema_version": 2, "receipt_source_generation": 1,
            "receipt_digest": "receipt-1", "receipt_verifier_revision": "verifier-1", "actual_plan_verified": true
        ]
        let result: [String: Any] = [
            "schema_version": 1, "model": original.model, "operation": original.operation,
            "output": "/tmp/fixture.png", "width": 512, "height": 512, "steps": 4,
            "seed": 42, "warmup": false, "public_streaming": summary
        ]
        func validate(_ value: [String: Any], request: NativeRequestV2 = bound) throws {
            try resolution.validateResult(JSONSerialization.data(withJSONObject: value), request: request)
        }
        try validate(result)
        try rejects { try validate(result, request: original) }
        try rejects { try validate([:]) }
        // Every bound identity must matter, including unknown schema versions,
        // numeric booleans, empty receipts and an unproven source generation.
        for (key, value) in summary {
            var changed = summary
            if value is String { changed[key] = "" }
            else { changed[key] = -1 }
            var bad = result; bad["public_streaming"] = changed
            try rejects { try validate(bad) }
            changed = summary; changed.removeValue(forKey: key)
            bad["public_streaming"] = changed
            try rejects { try validate(bad) }
        }
        for invalid: Any in [false, 1, "true"] {
            var changed = summary; changed["actual_plan_verified"] = invalid
            var bad = result; bad["public_streaming"] = changed
            try rejects { try validate(bad) }
        }
        for key in ["model", "operation", "output", "width", "height", "steps", "seed", "schema_version", "warmup"] {
            var changed = result
            changed[key] = result[key] is String ? "wrong" : -1
            try rejects { try validate(changed) }
        }
        var v3 = summary; v3["receipt_schema_version"] = 3
        var multiStage = result; multiStage["public_streaming"] = v3
        try validate(multiStage)
        print("PASS streaming App transaction: frozen selector, result correlation, native verified identity, strict types and receipt versions")
    }
}
