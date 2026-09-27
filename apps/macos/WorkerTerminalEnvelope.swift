import Foundation
import CoreFoundation

/// Validates worker messages against the submitted intent and a caller-pinned
/// runtime identity. Artifact bytes/decode/publication remain a separate check.
enum WorkerTerminalEnvelope {
    enum Operation { case query, generate }
    struct Artifact: Decodable, Sendable {
        let path: String
        let size: UInt64
        let sha256: String
    }
    struct Verified: Sendable {
        let status: String
        let resolution: NativeStreamingResolution?
        let result: Data?
        let artifact: Artifact?
        let errorCode: String?
        let errorMessage: String?
    }
    private static func invalid() -> NativeFailure {
        NativeFailure(message: "worker_terminal_invalid: 工作进程结果与本次请求不匹配。")
    }
    static func validate(_ data: Data, input: Data, request: NativeRequestV2,
                         runtimeFingerprint: String, operation: Operation,
                         exitCode: Int32) throws -> Verified {
        guard data.count <= 4 << 20, input.count <= 1 << 20, !runtimeFingerprint.isEmpty,
              let wire = try JSONSerialization.jsonObject(with: input) as? [String: Any],
              let inputVersion = wire["protocol_version"] as? NSNumber,
              CFGetTypeID(inputVersion) != CFBooleanGetTypeID(), inputVersion == 1,
              let submitted = wire["native_request_v2"] as? [String: Any],
              let expected = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any],
              NSDictionary(dictionary: submitted).isEqual(to: expected),
              let value = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = value["protocol_version"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              let status = value["status"] as? String,
              value["actual_container"] as? String == "cli_worker",
              value["runtime_fingerprint"] as? String == runtimeFingerprint else { throw invalid() }
        let digest = try WorkerRequestEnvelope.requestDigest(submitted)
        guard wire["request_digest"] as? String == digest,
              value["request_digest"] as? String == digest else { throw invalid() }
        for key in ["job_id", "request_id"] {
            guard let expectedID = wire[key] as? String,
                  UUID(uuidString: expectedID)?.uuidString.lowercased() == expectedID,
                  value[key] as? String == expectedID else { throw invalid() }
        }
        if status == "error" || status == "cancelled" {
            guard exitCode == (status == "cancelled" ? 2 : 1),
                  let error = value["error"] as? [String: Any],
                  let code = error["code"] as? String, !code.isEmpty,
                  let message = error["message"] as? String, !message.isEmpty,
                  status != "cancelled" || code == "worker_cancelled",
                  value["result"] == nil, value["resolution"] == nil else { throw invalid() }
            for key in ["resolution_digest", "record_digest", "layout_digest", "artifact", "public_streaming_summary"] {
                guard value[key] is NSNull else { throw invalid() }
            }
            return Verified(status: status, resolution: nil, result: nil, artifact: nil,
                            errorCode: code, errorMessage: message)
        }
        guard exitCode == 0, value["error"] is NSNull,
              status == (operation == .query ? "resolved" : "succeeded"),
              let rawResolution = value["resolution"] as? [String: Any] else { throw invalid() }
        let resolution = try JSONDecoder().decode(NativeStreamingResolution.self,
            from: JSONSerialization.data(withJSONObject: rawResolution))
        let bound = try resolution.binding(request)
        guard resolution.selection.execution_container == "cli_worker",
              value["resolution_digest"] as? String == resolution.resolution_digest,
              value["record_digest"] as? String == resolution.selection.record_digest,
              value["layout_digest"] as? String == resolution.selection.layout_digest else { throw invalid() }
        if operation == .query {
            guard value["artifact"] is NSNull, value["public_streaming_summary"] is NSNull,
                  value["result"] == nil else { throw invalid() }
            return Verified(status: status, resolution: resolution, result: nil, artifact: nil,
                            errorCode: nil, errorMessage: nil)
        }
        guard let rawResult = value["result"] as? [String: Any],
              let summary = value["public_streaming_summary"] as? [String: Any],
              let actualSummary = rawResult["public_streaming"] as? [String: Any],
              NSDictionary(dictionary: summary).isEqual(to: actualSummary),
              let rawArtifact = value["artifact"] as? [String: Any] else { throw invalid() }
        let result = try JSONSerialization.data(withJSONObject: rawResult)
        try resolution.validateResult(result, request: bound)
        let artifact = try JSONDecoder().decode(Artifact.self,
            from: JSONSerialization.data(withJSONObject: rawArtifact))
        guard bound.outputs.count == 1, artifact.path == bound.outputs[0].path,
              artifact.size > 0, artifact.sha256.utf8.count == 64,
              artifact.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw invalid() }
        return Verified(status: status, resolution: resolution, result: result, artifact: artifact,
                        errorCode: nil, errorMessage: nil)
    }
}
