import Foundation
import CryptoKit

/// Wire intent only. A digest correlates a request; it never grants execution authority.
enum WorkerRequestEnvelope {
    enum Failure: Error { case invalidRequest, invalidInstallation, tooLarge }
    static func requestDigest(_ request: [String: Any]) throws -> String {
        let canonical = try JSONSerialization.data(withJSONObject: request, options: [.sortedKeys, .withoutEscapingSlashes])
        var bytes = Data("tc-worker-request-v1\n".utf8)
        bytes.append(canonical)
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    static func encode(jobID: UUID, requestID: UUID, installation: URL, nativeRequest: Data) throws -> Data {
        guard nativeRequest.count <= 1 << 20 else { throw Failure.tooLarge }
        guard installation.isFileURL, !installation.path.contains("\0") else { throw Failure.invalidInstallation }
        guard let request = try JSONSerialization.jsonObject(with: nativeRequest) as? [String: Any] else {
            throw Failure.invalidRequest
        }
        let envelope: [String: Any] = ["protocol_version": 1, "job_id": jobID.uuidString.lowercased(),
            "request_id": requestID.uuidString.lowercased(), "request_digest": try requestDigest(request),
            "model_installation_ref": installation.standardizedFileURL.path, "native_request_v2": request]
        let result = try JSONSerialization.data(withJSONObject: envelope, options: [.sortedKeys, .withoutEscapingSlashes])
        guard result.count <= 1 << 20 else { throw Failure.tooLarge }
        return result
    }
}
